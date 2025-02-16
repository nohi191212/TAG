import os
import sys

import warnings
warnings.filterwarnings("ignore")

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)
    
import json
import argparse
import pathlib
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/PyTorchLightning/pytorch-lightning/issues/11663
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy

from transformers import BartTokenizer, BartModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    BartForConditionalGeneration

from tag.lib.dataset import load_aokvqa, AokvqaEmbeddingsDataModule, AokvqaEmbeddingsDataset
from tag.lib.model import ClipBartTAG, ClipBartTAGFinetune

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, default='datasets/aokvqa/', required=False,
                        dest='aokvqa_dir')
    parser.add_argument('--vocab', type=argparse.FileType('r'),
                        default='datasets/aokvqa/large_vocab_train.csv', required=False)
    parser.add_argument('--log-dir', type=pathlib.Path, default='logs/', dest='log_dir',
                        required=False)
    parser.add_argument('--backbone', type=str, choices=['clip', 'resnet', 'bert'], default='clip', required=False)
    parser.add_argument('--clip-model-type', type=str,
                        choices=['RN50', 'RN50x4', 'RN50x16', 'RN50x64', 'RN101', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14',
                                 'ViT-L/14@336px'],
                        dest='clip_model_type', default='ViT-L/14@336px', required=('clip' in sys.argv))
    parser.add_argument('--train-features', type=pathlib.Path,
                        default='datasets/aokvqa/features/rationale_336px_train', required=False,
                        dest='train_features')
    parser.add_argument('--val-features', type=pathlib.Path,
                        default='datasets/aokvqa/features/rationale_336px_val', required=False,
                        dest='val_features')
    parser.add_argument('--vocab-features', type=pathlib.Path,
                        default='datasets/aokvqa/features/clip-ViT-L-14-336px-large-vocab.pt', required=False,
                        dest='vocab_features')
    parser.add_argument('--objective', type=str, choices=['classifier', 'contrastive'], default='contrastive',
                        required=False)
    parser.add_argument('--bart_path', type=str, default='BART/bart-large', required=False)
    parser.add_argument('--inputs', nargs='+', type=str, choices=['question', 'image'], default='image', required=False)
    # Defaults
    parser.add_argument('--bs', type=int, default=2, dest='batch_size')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--task', type=str, choices=['MC', 'DA'], default='MC', required=False)
    
    
    args = parser.parse_args()

    pl.seed_everything(1)
    vocab = args.vocab.read().splitlines()

    ## Data loading

    dm = AokvqaEmbeddingsDataModule(
        args.aokvqa_dir,
        args.train_features,
        args.val_features,
        args.objective,
        args.backbone,
        args.inputs,
        vocab,
        args.vocab_features,
        batch_size=args.batch_size,
        num_workers=16
    )

    aokvqa_set = load_aokvqa(args.aokvqa_dir, 'train')
    len_train = len(aokvqa_set)
    steps_per_epoch = len_train // args.batch_size
    print("len_train, steps_per_epoch: ", len_train, steps_per_epoch)
    ## Model definition

    model = ClipBartTAG(
        args.bart_path,
        args.objective,
        args.backbone,
        args.clip_model_type,
        args.inputs,
        len(vocab),
        args.lr,
        args.epochs,
        steps_per_epoch
    )
    ## Training and testing loops
    logger = pl.loggers.TensorBoardLogger(
        args.log_dir,
        name=args.name
    )

    trainer = pl.Trainer(
        logger=logger,
        devices=args.gpus,
        max_epochs=args.epochs,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                filename="{epoch:02d}-{val_acc:.4f}",
                mode="max"
            )
        ],
    )

    trainer.fit(model, dm)
    
    
if __name__ == '__main__':
    main()