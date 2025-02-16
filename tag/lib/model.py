import os
import sys

sys.path.append("..")
import json
import argparse
import pathlib
import random

import clip

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import pytorch_lightning as pl

from transformers import BartTokenizer, BartModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    BartForConditionalGeneration

from .dataset import shift_tokens_right


class ClipBartTAG(pl.LightningModule):
    def __init__(self, bart_path, objective, backbone, clip_model_type, inputs, vocab_len, lr=0.001, epochs=100,
                 steps_per_epoch=100, bsz=32):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        if self.hparams.backbone == 'clip':
            clip_dim = {
                'RN50': 1024,
                'RN50x4': 640,
                'RN50x16': 768,
                'RN50x64': 1024,
                'RN101': 512,
                'ViT-B/32': 512,
                'ViT-B/16': 512,
                'ViT-L/14': 768,
                'ViT-L/14@336px': 768,
            }[clip_model_type]
            emb_dim = clip_dim * len(inputs)
        elif self.hparams.backbone == 'resnet':
            emb_dim = 2048
        elif self.hparams.backbone == 'bert':
            emb_dim = 768

        if self.hparams.objective == 'classifier':
            out_dim = vocab_len
        elif self.hparams.objective == 'contrastive':
            out_dim = clip_dim

        bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
        if 'large' in bart_path:
            text_dim = 1024
        else:
            text_dim = 768
        self.encoder = bart_model.model.encoder
        self.decoder = bart_model.model.decoder
        self.lm_head = bart_model.lm_head
        
        # project the BART text embeddings to CLIP latent space
        self.proj_text_bart2clip = nn.Linear(text_dim, out_dim)
        # project the CLIP image embeddings to BART latent space
        self.proj_image_clip2bart = nn.Linear(out_dim, text_dim)
        
        # for Cross-model Affinity Matrix (M), i.e. the attention
        self.text_linear = nn.Linear(text_dim, text_dim)
        self.dim2weight = nn.Linear(text_dim, 1)
        
        # temperature for contrastive loss
        self.temperature = 0.02


    def image_qo_attn(self, image_embd_d2, qo_embd_d2, qo_mask):
        """ Cross-modal affinity matrix (M) in paper Eq. (13) 
        
        Args:
            image_embd_d2 : # [bsz, 1, dim_bart]
            qo_embd_d2 : # [bsz, max_qo_n_tokens, dim_bart]
            qo_mask : # [bsz, max_qo_n_tokens]
            
        Return:
            qo_attn_logit : # [bsz, 1, max_qo_n_tokens]
        """
        
        qo_embd_d2 = self.text_linear(qo_embd_d2) # [bsz, max_qo_n_tokens, dim_bart]
        item = image_embd_d2.unsqueeze(2) + qo_embd_d2.unsqueeze(1) \
            # [bsz, 1, 1, dim_bart] * [bsz, 1, max_qo_n_tokens, dim_bart] = [bsz, 1, max_qo_n_tokens, dim_bart]
        qo_attn_logit = self.dim2weight(torch.tanh(item)).squeeze(-1) # [bsz, 1, max_qo_n_tokens]
        qo_mask = qo_mask.unsqueeze(1) # [bsz, 1, max_qo_n_tokens]
        qo_attn_logit = qo_attn_logit.masked_fill(qo_mask == 0, -1e30)
        return qo_attn_logit


    def encode_and_project_qa(self, qa_ids, qa_ids_mask):
        """
        Args:
            qa_ids : [bsz*n_o, max_qa_n_tokens], value the index of token in vocab
            qa_ids_mask : [bsz*n_o, max_qa_n_tokens], value 0 or 1

        Returns:
            sentence_embd_d1: [bsz*n_o, dim_clip], the embedding of the sentence "Question + Answer" in CLIP latent space
            sentence_embds_d2: [bsz*n_o, max_qa_n_tokens, dim_bart], the embedding of each token in the sentence "Question + Answer" in BART latent space
        """
        # language model encode all "Question + Answer"s to BART latent space
        sentence_embds_d2 = self.encoder(qa_ids, qa_ids_mask).last_hidden_state # [bsz*n_o, max_qa_n_tokens, dim_bart]
        
        # average pooling over all tokens, to represent the whole sentence "Question + Answer"
        mean_sentence_embd = self.mean(sentence_embds_d2, qa_ids_mask) # [bsz*n_o, dim_bart]
        
        # project the language model to CLIP latent space
        sentence_embd_d1 = self.proj_text_bart2clip(mean_sentence_embd) # [bsz*n_o, dim_clip]
        
        return sentence_embd_d1, sentence_embds_d2


    def forward_rationale_generation(self, image_clip_feat, encoder_input, encoder_mask, decoder_input, decoder_mask):
        """
        Args:
            image_clip_feat : [bsz, dim_clip]
            encoder_input : [bsz, max_qo_n_tokens], the input of the BART encoder is the "Question + Options" (ids)
            encoder_mask : [bsz, max_qo_n_tokens], value 0 or 1
            decoder_input : [bsz, max_r_n_tokens], the input of the BART decoder is the "Rationale" (ids)
            decoder_mask : [bsz, max_r_n_tokens], value 0 or 1

        Returns:
            loss_rg : loss of Rationale Generation (RG)
        """
        # language model encode the "Question + Options" (ids) to BART latent space
        qo_embd_d2 = self.encoder(encoder_input, encoder_mask).last_hidden_state \
            # [bsz, max_qo_n_tokens, dim_bart]

        # cross modal fusion
        image_embd_d2 = self.proj_image_clip2bart(image_clip_feat).unsqueeze(1) # [bsz, 1, dim_bart]
        
        # the shape of overall embedding C is [bsz, 3+max_qo_n_tokens, dim_bart]
        # so its attention mask is [bsz, 3+max_qo_n_tokens]
        bsz = qo_embd_d2.size(0)
        attention_mask = torch.cat((torch.ones(bsz, 3, device=self.device), encoder_mask), dim=1) # [bsz, 3+max_qo_n_tokens]

        # image to "Question + Options" attention
        attn_logit = self.image_qo_attn(image_embd_d2, qo_embd_d2, encoder_mask)
        image_text_attn, text_image_attn = attn_logit.softmax(-1), attn_logit.softmax(1) # [bsz, 1, max_qo_n_tokens]
        overall_qo_embd_d2 = torch.matmul(image_text_attn, qo_embd_d2) # [bsz, 1, dim_bart]
        cross_modal_fusion = image_embd_d2 * overall_qo_embd_d2 # [bsz, 1, dim_bart]
        
        # align witht the symbols in the paper's formulas.
        C_part_1 = image_embd_d2        # [bsz, 1, dim_bart]
        C_part_2 = cross_modal_fusion   # [bsz, 1, dim_bart]
        C_part_3 = overall_qo_embd_d2   # [bsz, 1, dim_bart]
        C_part_4 = qo_embd_d2           # [bsz, max_qo_n_tokens, dim_bart]

        # overall embedding C, [3+max_qo_n_tokens, dim_bart]
        encoder_outputs = torch.cat((C_part_1, C_part_2, C_part_3, C_part_4), dim=1) \
            # [bsz, 3+max_qo_n_tokens, dim_bart]
        
        # decoding the overall embedding C and the "Rationale" (ids)
        input_ids = shift_tokens_right(
            input_ids=decoder_input, # [bsz, max_r_n_tokens]
            pad_token_id=1, 
            decoder_start_token_id=2 
        )  # [bsz, max_r_n_tokens]
        outputs = self.decoder(
            input_ids=input_ids, # [bsz, max_r_n_tokens]
            attention_mask=decoder_mask, # [bsz, max_r_n_tokens]
            encoder_hidden_states=encoder_outputs, # [bsz, 3+max_qo_n_tokens, dim_bart]
            encoder_attention_mask=attention_mask, # [bsz, 3+max_qo_n_tokens]
        )
        """
        outputs: transformers.modeling_outputs.BaseModelOutputWithPastAndCrossAttentions (iterable)
            idx_0: [bsz, max_r_n_tokens, dim_bart]
            idx_1: tuple of length '??'
                idx_1_0: [bsz, n_head, ??, dim_bard // n_head]
                ...
        """
        
        # language model head decode the outputs embeddings to token logits
        lm_logits = self.lm_head(outputs[0]) # [bsz, max_r_n_tokens, vocab_size]
        vocab_size = lm_logits.size(-1)
        
        # calculate Rationale Generation (RG) loss
        lm_logits = lm_logits.view(-1, vocab_size) # [bsz*max_r_n_tokens, vocab_size], logit of each token-id
        rationale_ids = decoder_input.view(-1) # [bsz*max_r_n_tokens] (ids)
        loss_rg = F.cross_entropy(lm_logits, rationale_ids)
        
        return loss_rg
    
    
    def mean(self, x, mask):
        dim = x.size(-1)
        x_after = x * mask.unsqueeze(-1).repeat(1, 1, dim)
        x_sum = x_after.sum(1)
        mask_sum = mask.sum(1)
        y = x_sum / mask_sum.unsqueeze(-1).repeat(1, dim)
        return y
    
    
    def loss_v2e(self, v2e, gt_idxs_global):
        """
        Args:
            v2e : [bsz, bsz*n_o], image to all sentences attention, sentence="Question + Answer"
            gt_idxs_global : [bsz], the global index of the correct option in (bsz*n_o)

        Returns:
            loss_v2e
        """
        # contrastive loss between v and all sentences
        n_o = v2e.shape[1] // v2e.shape[0]
        # [bsz, bsz*n_o] -> [bsz, n_o]
        v2e = torch.stack([v2e[i, i*n_o:(i+1)*n_o] for i in range(v2e.shape[0])]) # [bsz, n_o]
        gt_idxs_global = gt_idxs_global % n_o # [bsz]
        return F.cross_entropy(v2e, gt_idxs_global)


    def loss_e2s(self, e, s):
        """
        Args:
            e : [bsz*n_o, dim_clip], BART sentence embedding, projected to CLIP latent space
            s : [bsz*n_o, dim_clip], CLIP sentence embedding, sentence="Question + Answer"
        Returns:
            loss_e2s
        """
        t2t = ((e @ s.T) / self.temperature).softmax(dim=-1) # BART-all to CLIP-all
        t2t_gt = torch.arange(0, s.shape[0], dtype=torch.int64, device=self.device) # [bsz*n_0]
        return F.cross_entropy(t2t, t2t_gt)


    def loss_e2v(self, e, v, gt_idxs_global):
        """
        Args:
            e : [bsz*n_o, dim_clip], BART sentence embedding, projected to CLIP latent space
            v : [bsz, dim_clip], CLIP image embedding
            gt_idxs_global : [bsz], value the global index of the correct option in (bsz*n_o)
        Returns:
            loss_e2v
        """
        # contrastive loss between e_pos (1/4) and v
        e_pos = e[gt_idxs_global] # [bsz, dim_clip], positive sentence embedding of image_i
        e2v = ((e_pos @ v.T) / self.temperature).softmax(dim=-1) # t_gt @ v.T -> [bsz, bsz]
        indices = torch.arange(0, v.shape[0], dtype=torch.int64, device=self.device)
        return F.cross_entropy(e2v, indices)


    def get_mc_acc(self, v2e, gt_idx):
        if v2e.argmax().item() == gt_idx[0]:
            return 1.0
        else:
            return 0.0


    def forward(self, batch, is_training: bool):
        """
        dim_clip: 768,  dim_bart: 1024,  n_o: 4
        id_start: 0,  id_end: 1,  
        r = rationale,  qo = q + options,  qa = q + right option
        
        Args:
            image_clip_feat : [bsz, dim_clip]
            text_clip_feat : [bsz, n_o, dim_clip]
            gt_idxs : [bsz] # the index of the correct choice, e.g. [0, 1, 2, 0] for bsz=4
            qa_ids: [bsz, n_o, max_qa_n_tokens], value the index of token in vocab
            qa_ids_mask: [bsz, n_o, max_qa_n_tokens],  value 0 or 1
            qo_ids: [bsz, max_qo_n_tokens], value the index of token in vocab
            qo_ids_mask: [bsz, max_qo_n_tokens], value 0 or 1
            r_ids: [bsz, max_r_n_tokens], value the index of token in vocab
            r_ids_mask: [bsz, max_r_n_tokens], value 0 or 1
        Returns:
            _type_: _description_
        """
        image_clip_feat, text_clip_feat, gt_idxs, qa_ids, qa_ids_mask, qo_ids, qo_ids_mask, r_ids, r_ids_mask = batch
        
        # batch size
        bsz = image_clip_feat.size(0)
        # the input of the BART encoder is the "Question + Options" (ids)
        encoder_input = qo_ids
        encoder_mask = qo_ids_mask
        # the input of the BART decoder is the "Rationale" (ids)
        decoder_input = r_ids
        decoder_mask = r_ids_mask

        # project the sentence "Question + Answer" to CLIP latent space
        qa_ids = qa_ids.reshape(-1, qa_ids.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        qa_ids_mask = qa_ids_mask.reshape(-1, qa_ids_mask.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        sentence_embd_d1, sentence_embds_d2 = self.encode_and_project_qa(qa_ids, qa_ids_mask)
        sentence_embd_d1 = F.normalize(sentence_embd_d1, dim=-1) # [bsz*n_choices, dim_clip]
        
        # Align with the symbols in the paper's formulas.
        v = image_clip_feat # [bsz, dim_clip]
        s = text_clip_feat # [bsz, n_o, dim_clip]
        s = s.reshape(-1, s.size(-1)) # [bsz*n_o, dim_clip]
        e = sentence_embd_d1 # [bsz*n_o, dim_clip]
        
        # clip_image_feat(v) to bart_text_feat(e) attention
        v2e = ((v @ e.T) / self.temperature).softmax(dim=-1) # v @ e.T -> [bsz, bsz*n_o]
        
        # matrix: An upper triangular matrix with all diagonal elements being 0 and all other 
        # elements in the upper triangle being 4. 'matrix' is to calculate the global index of the correct choice.
        matrix = torch.triu(torch.ones((bsz, bsz), device=self.device), diagonal=1) * 4
        gt_idxs_global = torch.tensor((gt_idxs + matrix.sum(0)), dtype=torch.int64) \
            # [bsz], value the global index of the correct option in (bsz*n_o)
        
        """ Loss TA: Triple Alignment Loss (TA) """
        # v: [bsz, dim_clip], CLIP image embedding
        # s: [bsz*n_o, dim_clip], CLIP sentence embedding, sentence="Question + Answer" 
        # e: [bsz*n_o, dim_clip], BART sentence embedding, projected to CLIP latent space
        
        # pull the positive sentence embedding of image_i to the CLIP image embedding of image_i
        # push the positive sentence embedding of image_i away from the CLIP image embedding of image_j (i!=j)
        loss_e2v = self.loss_e2v(e, v, gt_idxs_global)
        
        # pull the positive sentence embedding of image_i to the CLIP image embedding of image_i
        # push the negative sentence embedding of image_i away from the CLIP image embedding of image_i
        # v2e: [bsz, bsz*n_o], gt_idxs_global: [bsz] 
        loss_v2e = self.loss_v2e(v2e, gt_idxs_global)
        
        # pull the BART text embedding of sentence_i to the CLIP text embedding of sentence_i
        # push the BART text embedding of sentence_i away from the CLIP text embedding of sentence_j (i!=j)
        loss_e2s = self.loss_e2s(e, s)

        # Loss RG: Rationale Generation (RG)
        if is_training:
            # only calculate the loss when training
            loss_rg = self.forward_rationale_generation(v, encoder_input, encoder_mask, decoder_input, decoder_mask)
        else:
            loss_rg = 0.0

        # 1 1 2 0.5
        loss = 1.0 * loss_e2v + 1.0 * loss_v2e + 2.0 * loss_e2s + 0.1 * loss_rg
        # acc
        indices = torch.arange(0, bsz, dtype=torch.int64, device=self.device)
        if is_training:
            # when training, the accuracy is calculated based on the v2e attention
            acc = torch.mean(v2e[indices, gt_idxs_global])
        else:
            # when validating, calculate the accuracy of the predicted choice
            acc = self.get_mc_acc(v2e, gt_idxs_global)
        return loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg


    def training_step(self, batch, batch_idx):
        loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg = self.forward(batch, is_training=True)
        self.log("train_loss", loss)
        self.log("train_loss/loss_e2v", loss_e2v)
        self.log("train_loss/loss_v2e", loss_v2e)
        self.log("train_loss/loss_e2s", loss_e2s)
        self.log("train_loss/loss_rg", loss_rg)
        self.log("train_acc", acc)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg = self.forward(batch, is_training=False)
        self.log("val_loss", loss)
        self.log("val_loss/loss_e2v", loss_e2v)
        self.log("val_loss/loss_v2e", loss_v2e)
        self.log("val_loss/loss_e2s", loss_e2s)
        self.log("val_loss/loss_rg", loss_rg)
        self.log("val_acc", acc)
        return loss


    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        for name, p in self.encoder.named_parameters():
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        encoder_params = filter(lambda x: x.requires_grad is not False, self.encoder.parameters())
        for name, p in self.decoder.named_parameters():
            # p.requires_grad = False
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        decoder_params = filter(lambda x: x.requires_grad is not False, self.decoder.parameters())
        optimizer = torch.optim.Adam([
            {'params': encoder_params, 'lr': self.lr * 0.1},
            {'params': decoder_params, 'lr': self.lr * 0.1},
            {'params': self.lm_head.parameters(), 'lr': self.lr},
            {'params': self.proj_text_bart2clip.parameters(), 'lr': self.lr},
            {'params': self.proj_image_clip2bart.parameters(), 'lr': self.lr},
            {'params': self.text_linear.parameters(), 'lr': self.lr},
            {'params': self.dim2weight.parameters(), 'lr': self.lr},
        ], weight_decay=0)

        steps = self.epochs * self.steps_per_epoch
        warmup_steps = 1 * self.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]




class ClipBartTAGFinetune(ClipBartTAG):
    def __init__(self, bart_path, objective, backbone, clip_model_type, inputs, vocab_len, lr=0.001, epochs=100,
                 steps_per_epoch=100, num_clip_finetune_layers=1, bsz=32):
        super(ClipBartTAG, self).__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.num_clip_finetune_layers = num_clip_finetune_layers

        if self.hparams.backbone == 'clip':
            clip_dim = {
                'RN50': 1024,
                'RN50x4': 640,
                'RN50x16': 768,
                'RN50x64': 1024,
                'RN101': 512,
                'ViT-B/32': 512,
                'ViT-B/16': 512,
                'ViT-L/14': 768,
                'ViT-L/14@336px': 768,
            }[clip_model_type]
            emb_dim = clip_dim * len(inputs)
        elif self.hparams.backbone == 'resnet':
            emb_dim = 2048
        elif self.hparams.backbone == 'bert':
            emb_dim = 768

        if self.hparams.objective == 'classifier':
            out_dim = vocab_len
        elif self.hparams.objective == 'contrastive':
            out_dim = clip_dim
            
        # CLIP: clip.visual,  clip.transformer, clip.token_embedding, clip.ln_final
        self.clip, self.preprocess = clip.load(clip_model_type, device='cuda')

        # BART
        bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
        if 'large' in bart_path:
            text_dim = 1024
        else:
            text_dim = 768
        self.encoder = bart_model.model.encoder
        self.decoder = bart_model.model.decoder
        self.lm_head = bart_model.lm_head
        
        # project the BART text embeddings to CLIP latent space
        self.proj_text_bart2clip = nn.Linear(text_dim, out_dim)
        # project the CLIP image embeddings to BART latent space
        self.proj_image_clip2bart = nn.Linear(out_dim, text_dim)
        
        # for Cross-model Affinity Matrix (M), i.e. the attention
        self.text_linear = nn.Linear(text_dim, text_dim)
        self.dim2weight = nn.Linear(text_dim, 1)
        
        # temperature for contrastive loss
        self.temperature = 0.02
        
    
    def loss_clip(self, image_clip_feat, text_clip_feat, gt_idxs_global):
        """
        Args:
            image_clip_feat : [bsz, dim_clip]
            text_clip_feat : [bsz, n_o, dim_clip]
            gt_idxs_global : [bsz]
        """
        text_clip_feat = text_clip_feat.reshape(-1, text_clip_feat.size(-1)) # [bsz*n_o, dim_clip]
        image2text = ((image_clip_feat @ text_clip_feat.T) / self.temperature).softmax(dim=-1) # [bsz, bsz*n_o]
        return F.cross_entropy(image2text, gt_idxs_global)

    def forward(self, batch, is_training: bool):
        """
        dim_clip: 768,  dim_bart: 1024,  n_o: 4
        id_start: 0,  id_end: 1,  
        r = rationale,  qo = q + options,  qa = q + right option
        
        Args:
            image : [bsz, 3, 336, 336]
            qa_clip_ids : [bsz, n_o, 77], value the index of token in vocab
            gt_idxs : [bsz] # the index of the correct choice, e.g. [0, 1, 2, 0] for bsz=4
            qa_ids: [bsz, n_o, max_qa_n_tokens], value the index of token in vocab
            qa_ids_mask: [bsz, n_o, max_qa_n_tokens],  value 0 or 1
            qo_ids: [bsz, max_qo_n_tokens], value the index of token in vocab
            qo_ids_mask: [bsz, max_qo_n_tokens], value 0 or 1
            r_ids: [bsz, max_r_n_tokens], value the index of token in vocab
            r_ids_mask: [bsz, max_r_n_tokens], value 0 or 1
        Returns:
            _type_: _description_
        """
        image, qa_clip_ids, gt_idxs, qa_ids, qa_ids_mask, qo_ids, qo_ids_mask, r_ids, r_ids_mask = batch
        
        # CLIP fine-tuning: encode the image features
        image_clip_feat = self.clip.encode_image(image).float()
        image_clip_feat = image_clip_feat / image_clip_feat.norm(dim=-1, keepdim=True) # [bsz, dim_clip]
        # CLIP fine-tuning: encode the text features
        qa_clip_ids = qa_clip_ids.reshape(-1, qa_clip_ids.size(-1)) # [bsz*n_o, 77]
        text_clip_feat = self.clip.encode_text(qa_clip_ids).float() # [bsz*n_o, dim_clip]
        text_clip_feat = text_clip_feat / text_clip_feat.norm(dim=-1, keepdim=True) # [bsz*n_o, dim_clip]
        text_clip_feat = text_clip_feat.reshape(image.size(0), -1, text_clip_feat.size(-1)) # [bsz, n_o, dim_clip]
        
        # batch size
        bsz = image_clip_feat.size(0)
        # the input of the BART encoder is the "Question + Options" (ids)
        encoder_input = qo_ids
        encoder_mask = qo_ids_mask
        # the input of the BART decoder is the "Rationale" (ids)
        decoder_input = r_ids
        decoder_mask = r_ids_mask

        # project the sentence "Question + Answer" to CLIP latent space
        qa_ids = qa_ids.reshape(-1, qa_ids.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        qa_ids_mask = qa_ids_mask.reshape(-1, qa_ids_mask.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        sentence_embd_d1, sentence_embds_d2 = self.encode_and_project_qa(qa_ids, qa_ids_mask)
        sentence_embd_d1 = F.normalize(sentence_embd_d1, dim=-1) # [bsz*n_choices, dim_clip]
        
        # Align with the symbols in the paper's formulas.
        v = image_clip_feat # [bsz, dim_clip]
        s = text_clip_feat # [bsz, n_o, dim_clip]
        s = s.reshape(-1, s.size(-1)) # [bsz*n_o, dim_clip]
        e = sentence_embd_d1 # [bsz*n_o, dim_clip]
        
        # clip_image_feat(v) to bart_text_feat(e) attention
        v2e = ((v @ e.T) / self.temperature).softmax(dim=-1) # v @ e.T -> [bsz, bsz*n_o]
        
        # matrix: An upper triangular matrix with all diagonal elements being 0 and all other 
        # elements in the upper triangle being 4. 'matrix' is to calculate the global index of the correct choice.
        matrix = torch.triu(torch.ones((bsz, bsz), device=self.device), diagonal=1) * 4
        gt_idxs_global = torch.tensor((gt_idxs + matrix.sum(0)), dtype=torch.int64) \
            # [bsz], value the global index of the correct option in (bsz*n_o)
        
        """ Loss TA: Triple Alignment Loss (TA) """
        # v: [bsz, dim_clip], CLIP image embedding
        # s: [bsz*n_o, dim_clip], CLIP sentence embedding, sentence="Question + Answer" 
        # e: [bsz*n_o, dim_clip], BART sentence embedding, projected to CLIP latent space
        
        # pull the positive sentence embedding of image_i to the CLIP image embedding of image_i
        # push the positive sentence embedding of image_i away from the CLIP image embedding of image_j (i!=j)
        loss_e2v = self.loss_e2v(e, v, gt_idxs_global)
        
        # pull the positive sentence embedding of image_i to the CLIP image embedding of image_i
        # push the negative sentence embedding of image_i away from the CLIP image embedding of image_i
        # v2e: [bsz, bsz*n_o], gt_idxs_global: [bsz] 
        loss_v2e = self.loss_v2e(v2e, gt_idxs_global)
        
        # pull the BART text embedding of sentence_i to the CLIP text embedding of sentence_i
        # push the BART text embedding of sentence_i away from the CLIP text embedding of sentence_j (i!=j)
        loss_e2s = self.loss_e2s(e, s)

        # Loss RG: Rationale Generation (RG)
        if is_training:
            # only calculate the loss when training
            loss_rg = self.forward_rationale_generation(v, encoder_input, encoder_mask, decoder_input, decoder_mask)
        else:
            loss_rg = 0.0
            
        # CLIP contrastive loss
        loss_clip = self.loss_clip(image_clip_feat, text_clip_feat, gt_idxs_global)

        # 1 1 2 0.5
        loss = 1.0 * loss_e2v + 1.0 * loss_v2e + 1.0 * loss_e2s + 0.1 * loss_rg + 0.5 * loss_clip
        # acc
        indices = torch.arange(0, bsz, dtype=torch.int64, device=self.device)
        if is_training:
            # when training, the accuracy is calculated based on the v2e attention
            acc = torch.mean(v2e[indices, gt_idxs_global])
        else:
            # when validating, calculate the accuracy of the predicted choice
            acc = self.get_mc_acc(v2e, gt_idxs_global)
        return loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg, loss_clip
    
    
    def forward_test(self, batch):
        """
        n_o=4, max_qa_n_tokens=20, max_qo_n_tokens=30, max_r_n_tokens=30
        Args:
            image : [bsz, 3, 336, 336]
            qa_ids: [bsz, n_o, max_qa_n_tokens], value the index of token in vocab
            qa_ids_mask: [bsz, n_o, max_qa_n_tokens],  value 0 or 1
        Returns:
            _type_: _description_
        """
        #image, qa_clip_ids, gt_idxs, qa_ids, qa_ids_mask, qo_ids, qo_ids_mask, r_ids, r_ids_mask = batch
        image, qa_ids, qa_ids_mask = batch
        
        # CLIP fine-tuning: encode the image features
        image_clip_feat = self.clip.encode_image(image).float()
        image_clip_feat = image_clip_feat / image_clip_feat.norm(dim=-1, keepdim=True) # [bsz, dim_clip]
        
        # batch size
        bsz = image_clip_feat.size(0)
        # # the input of the BART encoder is the "Question + Options" (ids)
        # encoder_input = qo_ids
        # encoder_mask = qo_ids_mask
        # # the input of the BART decoder is the "Rationale" (ids)
        # decoder_input = r_ids
        # decoder_mask = r_ids_mask

        # project the sentence "Question + Answer" to CLIP latent space
        qa_ids = qa_ids.reshape(-1, qa_ids.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        qa_ids_mask = qa_ids_mask.reshape(-1, qa_ids_mask.size(-1)) # [bsz*n_choices, max_qa_n_tokens]
        sentence_embd_d1, sentence_embds_d2 = self.encode_and_project_qa(qa_ids, qa_ids_mask)
        sentence_embd_d1 = F.normalize(sentence_embd_d1, dim=-1) # [bsz*n_choices, dim_clip]
        
        # Align with the symbols in the paper's formulas.
        v = image_clip_feat # [bsz, dim_clip]
        e = sentence_embd_d1 # [bsz*n_o, dim_clip]
        
        # clip_image_feat(v) to bart_text_feat(e) attention
        v2e = ((v @ e.T) / self.temperature).softmax(dim=-1) # v @ e.T -> [1, n_o]
        
        choice = v2e.softmax(dim=-1).argmax(dim=-1).item() # [1]
        
        return choice


    def training_step(self, batch, batch_idx):
        loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg, loss_clip = self.forward(batch, is_training=True)
        self.log("train_loss", loss)
        self.log("train_loss/loss_e2v", loss_e2v)
        self.log("train_loss/loss_v2e", loss_v2e)
        self.log("train_loss/loss_e2s", loss_e2s)
        self.log("train_loss/loss_rg", loss_rg)
        self.log("train_loss/loss_clip", loss_clip)
        self.log("train_acc", acc)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, loss_e2v, loss_v2e, loss_e2s, loss_rg, loss_clip = self.forward(batch, is_training=False)
        self.log("val_loss", loss)
        self.log("val_loss/loss_e2v", loss_e2v)
        self.log("val_loss/loss_v2e", loss_v2e)
        self.log("val_loss/loss_e2s", loss_e2s)
        self.log("val_loss/loss_rg", loss_rg)
        self.log("val_loss/loss_clip", loss_clip)
        self.log("val_acc", acc)
        return loss



    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        
        # CLIP image encoder
        trainable_visual_postproj = ['proj', 'ln_post.weight', 'ln_post.bias']
        trainable_transformer_layers = [f'transformer.resblocks.{layer_idx}' for layer_idx in range(23, -1, -1)]
        trainable_transformer_layers = trainable_transformer_layers[:self.num_clip_finetune_layers]
        for name, p in self.clip.visual.named_parameters():
            if (name in trainable_visual_postproj):
                p.requires_grad = True
            elif(any(name.startswith(layer) for layer in trainable_transformer_layers)):
                p.requires_grad = True
            else:
                p.requires_grad = False
        visual_params = filter(lambda x: x.requires_grad is not False, self.clip.visual.parameters())

        # CLIP text encoder
        trainable_transformer_layers = [f'resblocks.{layer_idx}' for layer_idx in range(11, -1, -1)]
        trainable_transformer_layers = trainable_transformer_layers[:self.num_clip_finetune_layers]
        for name, p in self.clip.transformer.named_parameters():
            if any(name.startswith(layer) for layer in trainable_transformer_layers):
                p.requires_grad = False
            else:
                p.requires_grad = False
        text_params = filter(lambda x: x.requires_grad is not False, self.clip.transformer.parameters())
        
                
        # blacklisted parameters
        self.clip.positional_embedding.requires_grad = False
        self.clip.logit_scale.requires_grad = False
        self.clip.token_embedding.requires_grad = False
        self.clip.token_embedding.weight.requires_grad = False
        self.clip.ln_final.weight.requires_grad = False
        self.clip.ln_final.bias.requires_grad = False

        # BART encoder
        for name, p in self.encoder.named_parameters():
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        encoder_params = filter(lambda x: x.requires_grad is not False, self.encoder.parameters())
        # BART decoder
        for name, p in self.decoder.named_parameters():
            # p.requires_grad = False
            if name.startswith('layers.11') is False:
                p.requires_grad = False
        decoder_params = filter(lambda x: x.requires_grad is not False, self.decoder.parameters())
        optimizer = torch.optim.Adam([
            # CLIP image encoder
            {'params': visual_params, 'lr': self.lr * 0.1},
            # CLIP text encoder
            {'params': text_params, 'lr': self.lr * 0.1},
            # BART encoder last layer
            {'params': encoder_params, 'lr': self.lr * 0.1},
            # BART decoder last layer
            {'params': decoder_params, 'lr': self.lr * 0.1},
            # other layers
            {'params': self.lm_head.parameters(), 'lr': self.lr},
            {'params': self.proj_text_bart2clip.parameters(), 'lr': self.lr},
            {'params': self.proj_image_clip2bart.parameters(), 'lr': self.lr},
            {'params': self.text_linear.parameters(), 'lr': self.lr},
            {'params': self.dim2weight.parameters(), 'lr': self.lr},
        ], weight_decay=0)

        steps = self.epochs * self.steps_per_epoch
        warmup_steps = 1 * self.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]




class ClipFinetune(pl.LightningModule):
    def __init__(self, clip_model_type, lr=0.00001, epochs=100,
                 steps_per_epoch=100, num_clip_finetune_layers=1, bsz=32):
        super().__init__()
        self.save_hyperparameters(ignore=['lr'])
        self.lr = lr
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.num_clip_finetune_layers = num_clip_finetune_layers
            
        # CLIP: clip.visual,  clip.transformer, clip.token_embedding, clip.ln_final
        self.clip, self.preprocess = clip.load(clip_model_type, device='cuda')
        
        # temperature for contrastive loss
        self.temperature = 0.02
    
    
    def get_mc_acc(self, v2e, gt_idx):
        if v2e.argmax().item() == gt_idx[0]:
            return 1.0
        else:
            return 0.0
    

    def forward(self, batch, is_training: bool):
        """
        dim_clip: 768,  dim_bart: 1024,  n_o: 4
        id_start: 0,  id_end: 1,  
        r = rationale,  qo = q + options,  qa = q + right option
        
        Args:
            image : [bsz, 3, 336, 336]
            qa_clip_ids : [bsz, n_o, 77], value the index of token in vocab
            gt_idxs : [bsz] # the index of the correct choice, e.g. [0, 1, 2, 0] for bsz=4
            qa_ids: [bsz, n_o, max_qa_n_tokens], value the index of token in vocab
            qa_ids_mask: [bsz, n_o, max_qa_n_tokens],  value 0 or 1
            qo_ids: [bsz, max_qo_n_tokens], value the index of token in vocab
            qo_ids_mask: [bsz, max_qo_n_tokens], value 0 or 1
            r_ids: [bsz, max_r_n_tokens], value the index of token in vocab
            r_ids_mask: [bsz, max_r_n_tokens], value 0 or 1
        Returns:
            _type_: _description_
        """
        image, qa_clip_ids, gt_idxs, qa_ids, qa_ids_mask, qo_ids, qo_ids_mask, r_ids, r_ids_mask = batch
        # 检查输入数据是否包含异常值
        if torch.isnan(image).any() or torch.isnan(qa_clip_ids).any() or torch.isnan(gt_idxs).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(image).any() or torch.isinf(qa_clip_ids).any() or torch.isinf(gt_idxs).any():
            raise ValueError("Input contains Inf values")
        # batch size
        bsz = image.size(0)
        
        # CLIP fine-tuning: encode the image features
        image_clip_feat = self.clip.encode_image(image).float()
        image_clip_feat = image_clip_feat / image_clip_feat.norm(dim=-1, keepdim=True) # [bsz, dim_clip]
        # CLIP fine-tuning: encode the text features
        qa_clip_ids = qa_clip_ids.reshape(-1, qa_clip_ids.size(-1)) # [bsz*n_o, 77]
        text_clip_feat = self.clip.encode_text(qa_clip_ids).float() # [bsz*n_o, dim_clip]
        text_clip_feat = text_clip_feat / text_clip_feat.norm(dim=-1, keepdim=True) # [bsz*n_o, dim_clip]
        text_clip_feat = text_clip_feat.reshape(image.size(0), -1, text_clip_feat.size(-1)) # [bsz, n_o, dim_clip]
        
        # matrix: An upper triangular matrix with all diagonal elements being 0 and all other 
        # elements in the upper triangle being 4. 'matrix' is to calculate the global index of the correct choice.
        
        # CLIP image2text loss
        text_clip_feat = text_clip_feat.reshape(-1, text_clip_feat.size(-1)) # [bsz*n_o, dim_clip]
        image2text = ((image_clip_feat @ text_clip_feat.T) / self.temperature).softmax(dim=-1) # [bsz, bsz*n_o]
        i2t = image2text
        matrix = torch.triu(torch.ones((bsz, bsz), device=self.device), diagonal=1) * 4
        gt_idxs_i2t = torch.tensor((gt_idxs + matrix.sum(0)), dtype=torch.int64) \
            # [bsz], value the global index of the correct option in (bsz*n_o)
        loss_clip_i2t = F.cross_entropy(i2t, gt_idxs_i2t)
        
        # CLIP text2image loss
        text2image = ((text_clip_feat @ image_clip_feat.T) / self.temperature).softmax(dim=-1) # [bsz*n_o, bsz]
        text2image = text2image[gt_idxs_i2t] # select the text features of the correct option # [bsz, bsz]
        t2i = text2image
        gt_idxs_t2i = torch.range(0, bsz-1, device=self.device, dtype=torch.int64) # [bsz-1]
        loss_clip_t2i = F.cross_entropy(t2i, gt_idxs_t2i)
        # loss_clip_t2i = 0.0

        # loss
        loss = (loss_clip_i2t + loss_clip_t2i) 
        
        # acc
        indices = torch.arange(0, bsz, dtype=torch.int64, device=self.device)
        if is_training:
            # when training, the accuracy is calculated based on the v2e attention
            acc = torch.mean(i2t[indices, gt_idxs_i2t])
        else:
            # when validating, calculate the accuracy of the predicted choice
            acc = self.get_mc_acc(i2t, gt_idxs_i2t)
        return loss, acc, loss_clip_i2t, loss_clip_t2i


    def training_step(self, batch, batch_idx):
        loss, acc, loss_clip_i2t, loss_clip_t2i = self.forward(batch, is_training=True)
        self.log("train_loss", loss)
        self.log("train_loss/loss_clip_i2t", loss_clip_i2t)
        self.log("train_loss/loss_clip_t2i", loss_clip_t2i)
        self.log("train_acc", acc)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, acc, loss_clip_i2t, loss_clip_t2i = self.forward(batch, is_training=False)
        self.log("train_loss", loss)
        self.log("train_loss/loss_clip_i2t", loss_clip_i2t)
        self.log("train_loss/loss_clip_t2i", loss_clip_t2i)
        self.log("val_acc", acc)
        return loss


    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        
        # CLIP image encoder
        trainable_visual_postproj = ['proj', 'ln_post.weight', 'ln_post.bias']
        trainable_transformer_layers = [f'transformer.resblocks.{layer_idx}' for layer_idx in range(23, -1, -1)]
        trainable_transformer_layers = trainable_transformer_layers[:self.num_clip_finetune_layers]
        for name, p in self.clip.visual.named_parameters():
            if (name in trainable_visual_postproj):
                p.requires_grad = True
            elif(any(name.startswith(layer) for layer in trainable_transformer_layers)):
                p.requires_grad = True
            else:
                p.requires_grad = False
        visual_params = filter(lambda x: x.requires_grad is not False, self.clip.visual.parameters())

        # CLIP text encoder
        trainable_transformer_layers = [f'resblocks.{layer_idx}' for layer_idx in range(11, -1, -1)]
        trainable_transformer_layers = trainable_transformer_layers[:self.num_clip_finetune_layers]
        for name, p in self.clip.transformer.named_parameters():
            if any(name.startswith(layer) for layer in trainable_transformer_layers):
                p.requires_grad = False
            else:
                p.requires_grad = False
        text_params = filter(lambda x: x.requires_grad is not False, self.clip.transformer.parameters())
        
        # CLIP projection layers
        trainable_proj = [] #['text_projection', 'ln_final.weight', 'ln_final.bias']
        for name, p in self.clip.named_parameters():
            if (name in trainable_proj is False):
                p.requires_grad = False
                
        # blacklisted parameters
        self.clip.positional_embedding.requires_grad = False
        self.clip.logit_scale.requires_grad = False
        self.clip.token_embedding.requires_grad = False
        self.clip.token_embedding.weight.requires_grad = False
        self.clip.ln_final.weight.requires_grad = False
        self.clip.ln_final.bias.requires_grad = False
        
        optimizer = torch.optim.Adam([
            # CLIP image encoder
            {'params': visual_params, 'lr': self.lr},
            # CLIP text encoder
            {'params': text_params, 'lr': self.lr},
            # CLIP projection layers
            {'params': self.clip.ln_final.parameters(), 'lr': self.lr},
            {'params': self.clip.text_projection, 'lr': self.lr},
        ], weight_decay=0)
        
        # check trainable parameters
        print("require_grad of CLIP parameters:")
        for name, p in self.clip.named_parameters():
            if p.requires_grad:
                print(name, p.requires_grad)

        steps = self.epochs * self.steps_per_epoch
        warmup_steps = 1 * self.steps_per_epoch
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]

