import os
from PIL import Image
from tqdm import tqdm
import argparse
import pathlib
import json
import torch
import clip

import os
import sys

import warnings
warnings.filterwarnings("ignore")

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

from tag.lib.model import ClipFinetune

from transformers import BartTokenizer, BartModel

parser = argparse.ArgumentParser()
parser.add_argument('--aokvqa-dir', type=pathlib.Path, default='datasets/aokvqa/', required=False, dest='aokvqa_dir')
parser.add_argument('--coco-dir', type=pathlib.Path, default='datasets/coco/', required=False, dest='coco_dir')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
parser.add_argument('--model-type', type=str, choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'], required=False, default='ViT-L/14@336px', dest='model_type')
parser.add_argument('--tokenizer-type', type=str, choices=['bart-base', 'bart-large'], dest='tokenizer_type')
parser.add_argument('--output_dir', type=pathlib.Path, required=True, dest='output_dir')
parser.add_argument('--use-finetuned-clip', action='store_true', dest='use_finetuned_clip')
parser.add_argument('--clip-model-path', type=str, required=False, dest='clip_model_path', default=None)
args = parser.parse_args()

# assert args.output_file.suffix == '.pt'

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

## Load dataset
print("Load dataset")
dataset = load_aokvqa(args.aokvqa_dir, args.split)

## Load model
print("Load model")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BartTokenizer.from_pretrained(f'BART/{args.tokenizer_type}')

if not args.use_finetuned_clip:
    model, preprocess = clip.load(args.model_type, device=device)
else:
    import pdb
    pdb.set_trace()
    model_ = ClipFinetune(args.model_type)
    ckpt = torch.load(args.clip_model_path)
    model_.load_state_dict(ckpt['state_dict'])
    preprocess = model_.preprocess
    model = model_.clip
    model = model.to(device)

for name, param in model.transformer.named_parameters():
    print(name)

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

## Encoding loop
print("Encoding loop")
with torch.no_grad():
    embeddings = {}
    for d in tqdm(dataset):
        img = Image.open(get_coco_path(args.split, d['image_id'], args.coco_dir))
        img = preprocess(img).unsqueeze(0).to(device)
        image_features = model.encode_image(img)  # [1, 512]

        q = d["question"]  # What type of race is this?
        choices = d['choices']
        if 'rationales' in d.keys():
            rationales = d['rationales']
        else:
            rationales = ['No rationale.']
        # direct_answer = d['direct_answers']
        qa_list = []
        bart_ids = []

        da_list = []
        da_bart_ids = []
        da_target = []

        r = 'Rationale: ' + rationales[0]
        encoder_input = 'Question: ' + q + ' Options: '
        for c in choices:
            encoder_input += c + ', '
            qa = 'Question: ' + q + ' Answer: ' + c  # Question: What is the boy on the right holding? Answer: mace
            qa_text = clip.tokenize(qa).to(device)  # [1, 77] [[49406, 4382(Question), 281(:), 8 tokens, 286(?), 4518(Answer), 281(:), 44869, 49407, 0]
            qa_text_features = model.encode_text(qa_text)  # [1, 768]
            qa_list.append(qa_text_features[0].float().cpu())
            bart_ids.append(tokenizer(qa, return_tensors="pt"))

        qa_list = torch.stack(qa_list, dim=0)
        qa_list /= qa_list.norm(dim=-1, keepdim=True) # [n_choice, 768]

        image = image_features[0].float().cpu()
        image /= image.norm(dim=-1, keepdim=True)

        # da_list = torch.stack(da_list, dim=0)
        # da_list /= da_list.norm(dim=-1, keepdim=True)

        embedding = {
            'qa_list': qa_list, # [n_choice, 768]
            'image': image, # [768]
            'bart_inputs': bart_ids, # list: 4 x {'input_ids': [1, qa_n_tokens], 'attention_mask': [1, qa_n_tokens]}
            'encoder_input': tokenizer(encoder_input, return_tensors="pt"), # {'input_ids': [1, qo_n_tokens], 'attention_mask': [1, qo_n_tokens]}
            'rationale': tokenizer(r, return_tensors="pt"), # {'input_ids': [1, r_n_tokens], 'attention_mask': [1, r_n_tokens]}
            # 'da_bart_inputs': da_bart_ids,
            # 'da_list': da_list,
            # 'da_target': da_target,
            # 'answer_index': answer_idx
        }
        torch.save(embedding, args.output_dir / (d['question_id'] + ".pt"))
    # torch.save(embeddings, args.output_dir)
