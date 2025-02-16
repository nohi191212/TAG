import os
import sys

import warnings
warnings.filterwarnings("ignore")

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)
import argparse
import pathlib
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import sentencepiece; import pytorch_lightning as pl; import clip

from tag.lib.model import ClipBartTAG
from load_aokvqa import load_aokvqa
from transformers import BartTokenizer
# from evaluation.remap_predictions import map_to_choices


parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=False, dest='aokvqa_dir')
parser.add_argument('--candidates-dir', default='update_files/candidates_aokvqa_val.json', type=pathlib.Path, required=False, dest='candidates_dir')
parser.add_argument('--features', type=pathlib.Path, required=True)
parser.add_argument('--out', type=argparse.FileType('w'), dest='output_file')
#
parser_weights = parser.add_mutually_exclusive_group(required=True)

parser_weights.add_argument('--ckpt', type=pathlib.Path, dest='checkpoint_path')

parser_weights.add_argument('--zero-shot', action='store_true', dest='clip_zero_shot')
parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], required=('--zero-shot' in sys.argv))
parser.add_argument('--vocab', type=argparse.FileType('r'))
parser.add_argument('--clip-model-type', type=str,
                    choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
                    dest='clip_model_type', required=('--zero-shot' in sys.argv and '--mc' in sys.argv))
#
args = parser.parse_args()

# tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
tokenizer = BartTokenizer.from_pretrained('/home/huangsiyong/model_zoo/huggingface/bart_large')
aokvqa_set = load_aokvqa(args.aokvqa_dir, args.split)
qid_to_topk = json.load(open(args.candidates_dir))
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.checkpoint_path is not None:
    classifier = ClipBartTAG.load_from_checkpoint(args.checkpoint_path)
    classifier.to(device)
    hp = classifier.hparams
elif args.clip_zero_shot:
    classifier = nn.Identity().to(device)
    hp = pl.utilities.AttributeDict(backbone='clip', clip_model_type=args.clip_model_type, objective='zero-shot', inputs=args.inputs)

# Load vocab, vocab features, clip
if hp.objective in ['contrastive', 'zero-shot']:
        clip_model = clip.load(hp.clip_model_type, device=device)[0]
        logit_scale = clip_model.logit_scale.exp().cpu()

## Prediction loop

predictions = {}
correct_1 = 0
correct_2 = 0
num = 0
with torch.no_grad():
    for o in tqdm(aokvqa_set):
        q = o['question_id']
        embedding = torch.load(args.features / (q + ".pt"))
        i = embedding['image']
        candidates = qid_to_topk[q]
        bart_inputs = []
        confidence = []
        # qa_list = []
        for index in range(8):
            c = candidates[index]['answer']
            confidence.append(candidates[index]['confidence'])
            qa = 'Question: ' + o["question"] + ' Answer: ' + c  # Question: What is the boy on the right holding? Answer: mace
            tokenizer(qa, return_tensors="pt")
            bart_inputs.append(tokenizer(qa, return_tensors="pt"))
            # qa_list.append(qa)
        ids = []
        mask = []
        for p in bart_inputs:
            ids.append(p["input_ids"].squeeze(0))
            mask.append(p["attention_mask"].squeeze(0))
        ids = pad_sequence(ids, batch_first=True, padding_value=1)
        mask = pad_sequence(mask, batch_first=True)
        ids = ids.reshape(-1, ids.size(-1)).to(device)
        mask = mask.reshape(-1, mask.size(-1)).to(device)
        t_after, t_ori = classifier(ids, mask)

        t_norm = F.normalize(t_after, dim=-1).cpu()
        x = (i.cpu() @ t_norm.t()) / 0.5
        x = x.softmax(dim=-1)

        confidence = torch.tensor(confidence)
        confidence = confidence.softmax(dim=-1)
        # print(x, confidence, (x+confidence)/2)
        ratio = 0.2
        x = ratio * x + (1-ratio) * confidence

        predictions[q] = candidates[x.argmax().item()]['answer']
        direct_answers = o['direct_answers']
        if o['difficult_direct_answer'] is False:
            num += 1
            num_match_1 = sum([candidates[0]['answer'] == da for da in direct_answers])
            num_match_2 = sum([candidates[x.argmax().item()]['answer'] == da for da in direct_answers])
#             print(o['choices'][o['correct_choice_idx']], o['choices'])
#             print("top10:", inputs_for_da[q]['answer'])
#             print("direct_answers:", direct_answers)
#             print("answer:", vocab[inputs_for_da[q]['answer_idx'][0]], vocab[answer_idx])
            vqa_acc_1 = min(1.0, num_match_1 / 3.0)
            vqa_acc_2 = min(1.0, num_match_2 / 3.0)
            correct_1 += vqa_acc_1
            correct_2 += vqa_acc_2
#
print("acc:", correct_1/num, correct_2/num)
json.dump(predictions, args.output_file)
