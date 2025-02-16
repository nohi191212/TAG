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

checkpoint_path = 'logs/bs32_336_large_1_1_2_0.1/version_0/checkpoints/epoch=75-val_acc=0.68.ckpt'
tokenizer = BartTokenizer.from_pretrained('BART/bart_large')

# load data
qid_to_topk = json.load(open('datasets/prophet/assets/candidates_okvqa.json'))
# with open('/home/huangsiyong//data/prophet/assets/answer_dict_aokvqa.json') as f:
#     d = json.load(f)
# print(d[0])
with open('datasets/prophet/datasets/okvqa/mscoco_val2014_annotations.json') as f:
    val_datasets_annotations = json.load(f)['annotations']

for val_a in val_datasets_annotations:
    image_id = val_a['image_id']
    question_id = val_a['question_id']  # 数字

#organize answer list
val_datasets = []
for val_a in val_datasets_annotations:
    multi_answers = []
    for ans in val_a['answers']:
        # multi_answers.append(ans['raw_answer'])
        multi_answers.append(ans['answer'])
    row = {'question_id': val_a['question_id'], 'direct_answers': multi_answers}
    val_datasets.append(row)

embeddings = torch.load('features/okvqa_val.pt')

device = "cuda" if torch.cuda.is_available() else "cpu"

classifier = ClipBartTAG.load_from_checkpoint(checkpoint_path)
classifier.to(device)

# # Load vocab, vocab features, clip
# if hp.objective in ['contrastive', 'zero-shot']:
#         clip_model = clip.load(hp.clip_model_type, device=device)[0]
#         logit_scale = clip_model.logit_scale.exp().cpu()

## Prediction loop
predictions = {}
correct_1 = 0
correct_2 = 0
num = 0
with torch.no_grad():
    for o in tqdm(val_datasets):
        q = o['question_id']
        embedding = embeddings[q]
        i = embedding['image']
        question = embedding['question']
        candidates = qid_to_topk[str(q)]
        bart_inputs = []
        confidence = []
        # qa_list = []
        for index in range(4):
            c = candidates[index]['answer']
            confidence.append(candidates[index]['confidence'])
            qa = 'Question: ' + question + ' Answer: ' + c  # Question: What is the boy on the right holding? Answer: mace
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
        x = (i.cpu() @ t_norm.t()) / 1.0
        x = x.softmax(dim=-1)

        confidence = torch.tensor(confidence)
        confidence = confidence.softmax(dim=-1)
        # print(x, confidence, (x+confidence)/2)
        ratio = 0.2
        x = ratio * x + (1-ratio) * confidence

        predictions[q] = candidates[x.argmax().item()]['answer']
        direct_answers = o['direct_answers']
        # if o['difficult_direct_answer'] is False:
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
# json.dump(predictions, args.output_file)
