import os
import json
import tqdm

import clip
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.distributed import DistributedSampler

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import pytorch_lightning as pl


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    f = open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json"))
    dataset = json.load(f)
    f.close()
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_coco2014_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2014", f"COCO_val2014_{image_id:012}.jpg")


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

"""                 Feature-based Dataset with Pre-Computed Embeddings                 """
"""                 Feature-based Dataset with Pre-Computed Embeddings                 """
"""                 Feature-based Dataset with Pre-Computed Embeddings                 """

class AokvqaEmbeddingsDataset(Dataset):
    def __init__(self, aokvqa_dir, split, input_features, objective, backbone, inputs, vocab, vocab_features):
        self.aokvqa_set = load_aokvqa(aokvqa_dir, split)
        self.path = input_features
        self.objective = objective
        self.vocab_len = len(vocab)

    def __getitem__(self, index):
        o = self.aokvqa_set[index]
        q = o['question_id']
        embedding = torch.load(self.path / (q + ".pt"))
        gt = o['correct_choice_idx'] # decoder训练的gt是正确选项的索引
        i = embedding['image']
        t = embedding['qa_list']
        bart_input = embedding['bart_inputs']
        encoder_input = embedding['encoder_input']
        decoder_input = embedding['rationale']
        
        return i, t, gt, bart_input, encoder_input, decoder_input

    def __len__(self):
        return len(self.aokvqa_set)
    

class AokvqaEmbeddingsDataModule(pl.LightningDataModule):
    def __init__(self, aokvqa_dir, train_features, val_features, objective, backbone, inputs, vocab, vocab_features,
                 batch_size=1, num_workers=0):
        super().__init__()
        self.aokvqa_dir = aokvqa_dir
        self.train_features = train_features
        self.val_features = val_features
        self.objective = objective
        self.backbone = backbone
        self.inputs = inputs
        self.vocab = vocab
        self.vocab_features = vocab_features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'train', self.train_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )
        self.val_dataset = AokvqaEmbeddingsDataset(
            self.aokvqa_dir, 'val', self.val_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=int(0.8 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=int(0.2 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )


def aokvqa_collate_fn(batch):
    image = []
    text = []
    gt = []
    ids = []
    mask = []
    encoder_input = []
    encoder_mask = []
    decoder_input = []
    decoder_mask = []
    for b in batch:
        i, t, c, bart_input, e, d = b
        image.append(i)
        text.append(t) # [4, 1024]
        gt.append(torch.tensor(c))

        encoder_input.append(e["input_ids"].squeeze(0))
        encoder_mask.append(e["attention_mask"].squeeze(0))
        decoder_input.append(d["input_ids"].squeeze(0))
        decoder_mask.append(d["attention_mask"].squeeze(0))

        for p in bart_input:
            ids.append(p["input_ids"].squeeze(0))
            mask.append(p["attention_mask"].squeeze(0))
    image = torch.stack(image)
    text = torch.stack(text, dim=0)
    gt = torch.stack(gt)

    ids = pad_sequence(ids, batch_first=True, padding_value=1)
    mask = pad_sequence(mask, batch_first=True)

    encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value=1)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True)
    decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=1)
    decoder_mask = pad_sequence(decoder_mask, batch_first=True)

    return image, text, gt, ids.reshape(-1, 4, ids.size(-1)), mask.reshape(-1, 4, mask.size(-1)), \
           encoder_input, encoder_mask, decoder_input, decoder_mask






"""                 Fine-tuning Dataset without Pre-Computed Embeddings                 """
"""                 Fine-tuning Dataset without Pre-Computed Embeddings                 """
"""                 Fine-tuning Dataset without Pre-Computed Embeddings                 """

from PIL import Image

# for fine-tuning
class AokvqaFineTuneDataset(Dataset):
    def __init__(self, aokvqa_dir, coco_dir, split, input_features, objective, 
                 backbone, inputs, vocab, vocab_features, preprocess):
        self.split = split
        self.coco_dir = coco_dir
        self.aokvqa_set = load_aokvqa(aokvqa_dir, split)
        self.path = input_features
        self.objective = objective
        self.vocab_len = len(vocab)
        self.preprocess = preprocess
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, index):
        o = self.aokvqa_set[index]
        
        # image
        image_id = o['image_id']
        i = Image.open(get_coco_path(self.split, image_id, self.coco_dir))
        i = self.preprocess(i)#.to(self.device) # [3, 336, 336]
        # it would be [b, 3, 336, 336] after applying 'aokvqa_collate_fn'
        
        # text
        question = o['question']
        choices = o['choices']
        qa_list = []
        for choice in choices:
            qa = 'Question: ' + question + ' Answer: ' + choice
            qa_text = clip.tokenize(qa)#.to(self.device) # [1, 77]
            qa_list.append(qa_text)
        t = torch.cat(qa_list, dim=0) # [4, 77]
        # it would be [b, 4, 77] after applying 'aokvqa_collate_fn'

        q = o['question_id']
        embedding = torch.load(self.path / (q + ".pt"))
        gt = o['correct_choice_idx']
        # i = embedding['image']
        # t = embedding['qa_list']
        
        bart_input = embedding['bart_inputs']
        encoder_input = embedding['encoder_input']
        decoder_input = embedding['rationale']
        return i, t, gt, bart_input, encoder_input, decoder_input

    def __len__(self):
        return len(self.aokvqa_set)


class AokvqaFineTuneDataModule(pl.LightningDataModule):
    def __init__(self, aokvqa_dir, coco_dir, train_features, val_features, objective, backbone, inputs, vocab, vocab_features,
                 batch_size=1, num_workers=0, image_preprocess_fn=None):
        super().__init__()
        self.aokvqa_dir = aokvqa_dir
        self.coco_dir = coco_dir
        self.train_features = train_features
        self.val_features = val_features
        self.objective = objective
        self.backbone = backbone
        self.inputs = inputs
        self.vocab = vocab
        self.vocab_features = vocab_features
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # CLIP image preprocess function
        self.preprocess = image_preprocess_fn

    def setup(self, stage=None):
        self.train_dataset = AokvqaFineTuneDataset(
            self.aokvqa_dir, self.coco_dir, 'train', self.train_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features, self.preprocess
        )
        self.val_dataset = AokvqaFineTuneDataset(
            self.aokvqa_dir, self.coco_dir, 'val', self.val_features, self.objective,
            self.backbone, self.inputs, self.vocab, self.vocab_features, self.preprocess
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=int(0.8 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1, shuffle=False,
            num_workers=int(0.2 * self.num_workers),
            persistent_workers=True,
            collate_fn=aokvqa_collate_fn
        )
        
        
def aokvqa_finetune_collate_fn(batch):
    image = []
    text = []
    gt = []
    ids = []
    mask = []
    encoder_input = []
    encoder_mask = []
    decoder_input = []
    decoder_mask = []
    for b in batch:
        i, t, c, bart_input, e, d = b
        image.append(i)
        text.append(t) # list of n_o strings
        gt.append(torch.tensor(c))

        encoder_input.append(e["input_ids"].squeeze(0))
        encoder_mask.append(e["attention_mask"].squeeze(0))
        decoder_input.append(d["input_ids"].squeeze(0))
        decoder_mask.append(d["attention_mask"].squeeze(0))

        for p in bart_input:
            ids.append(p["input_ids"].squeeze(0))
            mask.append(p["attention_mask"].squeeze(0))
    image = torch.stack(image)
    # text = torch.stack(text, dim=0)
    gt = torch.stack(gt)

    ids = pad_sequence(ids, batch_first=True, padding_value=1)
    mask = pad_sequence(mask, batch_first=True)

    encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value=1)
    encoder_mask = pad_sequence(encoder_mask, batch_first=True)
    decoder_input = pad_sequence(decoder_input, batch_first=True, padding_value=1)
    decoder_mask = pad_sequence(decoder_mask, batch_first=True)

    return image, text, gt, ids.reshape(-1, 4, ids.size(-1)), mask.reshape(-1, 4, mask.size(-1)), \
           encoder_input, encoder_mask, decoder_input, decoder_mask