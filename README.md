# TAG

Repository for **TAG: Triple Alignment with Rationale Generation for Knowledge-based Visual Question Answering**

Some of the code in this repository is borrowed from A-OKVQA's code repository [[A-OKVQA]](https://github.com/allenai/aokvqa))



## Abstract

Knowledge-based Visual Question Answering (VQA) involves answering questions based not only on the given image, but also on external knowledge. Existing methods for knowledge-based VQA can be broadly classified into two main categories: those that rely on external knowledge bases, and those that use Large Language Models (LLMs) as implicit knowledge engines. However, the former approach heavily relies on the quality of information retrieval, introducing additional information bias to the entire system. And the latter approach suffers from the extremely high computational cost and the loss of image information. To address these issues, we propose a novel framework called TAG that reformulates knowledge-based VQA as a contrastive learning problem. TAG framework adopts a **T**riple asymmetric paradigm to **A**lign embeddings and **G**enerate corresponding rationale. 
We innovatively propose a triple asymmetric paradigm, which aligns a lightweight text encoder to the image space with an extremely low training cost (**0.0152B** trainable parameters), and enhance its understanding ability on semantic granularity. 
TAG is both computation-efficient and effective, and we evaluate it on the knowledge-based VQA datasets, A-OKVQA, OK-VQA and VCR. The results show that TAG (**0.387B** parameters) achieves the state-of-the-art performance when compared to methods using less than 1B parameters. When compared to approaches combined with LLM (GPT-3 with **170B** parameters), TAG still shows competitive performance.




## Table of Contents

- [Getting started](#getting-started)
  - [Prepare environment](#prepare-environment)
  - [Prepare A-OKVQA dataset](#prepare-a-okvqa-dataset)
  - [Prepare OK-VQA dataset](#prepare-ok-vqa-dataset)
  - [Prepare VCR dataset](#prepare-vcr-dataset)
  - [Organize the directory](#organize-the-directory)

- [Experiment](#experiment)
  - [Extracting features](#extracting-features)
  - [Training](#training)
  - [Evaluation](#evaluation)






## Getting started

### Prepare environment

```bash
git clone https://github.com/nohi191212/tag.git

cd TAG

conda env create --name tag python=3.9
conda activate tag

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightening==2.0 clip openai-clip transformers sentencepiece tqdm
```

### Prepare A-OKVQA dataset

download A-OKVQA annotations:

```bash
export AOKVQA_DIR=~/datasets/aokvqa/
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}
```

download COCO2017 images:


```bash
export COCO_DIR=~/datasets/coco2017/
mkdir -p ${COCO_DIR}

for split in train val test; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip
```

### Prepare OK-VQA dataset

Please download the OK-VQA annotations and images at [VCR: Visual Commonsense Reasoning](https://visualcommonsense.com/download/).

Organize your directory following [this](#organize-the-directory).

### Prepare VCR dataset

Please download the VCR annotations and images at [VCR: Visual Commonsense Reasoning](https://visualcommonsense.com/download/).

Organize your directory following [this](#organize-the-directory).

### 

### Organize the directory

organized dataset folder should be like this:

```
>> TAG
	>> BART
		>> bart-large	   		<--- download from huggingface facebook/bart-large
	>> data_scripts
	>> datasets
		>> aokvqa
			>> features    		<--- mkdir
			- aokvqa_v1p0_test.json
			- aokvqa_v1p0_train.json
			- aokvqa_v1p0_val.json
			- large_vocab_train.csv
            		- specialized_vocab_train.csv
		>> coco2017
			>> annotations
			>> test2017
			>> train2017
			>> val2017
		>> coco2014
			>> annotations
			>> test2014
			>> train2014
			>> val2014
		>> vcr
			>> features    		<--- mkdir
			>> vcr1images
			- test.jsonl
			- train.jsonl
			- val.jsonl
	>> logs				   	<--- mkdir
	>> tag
```



## Experiment

### Extracting features

convert VCR to A-OKVQA style

```
python data_scripts/convert.py \
    --vcr_dir datasets/vcr \
    --split train
    
python data_scripts/convert.py \
    --vcr_dir datasets/vcr \
    --split val
```

vocab features

```
python data_scripts/encode_vocab_clip.py \
    --vocab datasets/aokvqa/large_vocab_train.csv \
    --model-type ViT-L/14@336px \
    --out clip-ViT-L-14-336px-large-vocab.pt
```

clip features

```
python data_scripts/extract_rationales.py \
    --aokvqa-dir datasets/aokvqa \
    --coco-dir datasets/coco2017 \
    --split train \
    --model-type ViT-L/14@336px \
    --tokenizer-type bart-large \
    --output_dir datasets/aokvqa/features/rationale_336px_train_finetune \
    --use-finetuned-clip \
    --clip-model-path YOUR-CKPT-PATH
```



If you want to extract features using fine-tuned CLIP: 

vocab features (finetuned clip)

```
python data_scripts/encode_vocab_clip.py \
    --vocab datasets/aokvqa/large_vocab_train.csv \
    --model-type ViT-L/14@336px \
    --out clip-ViT-L-14-336px-large-vocab-finetune.pt \
    --use-finetuned-clip \
    --clip-model-path YOUR-CKPT-PATH
```

clip features (finetune)

```
python data_scripts/extract_rationales.py \
    --aokvqa-dir datasets/aokvqa \
    --coco-dir datasets/coco2017 \
    --split train \
    --model-type ViT-L/14@336px \
    --tokenizer-type bart-large \
    --output_dir datasets/aokvqa/features/rationale_336px_train_finetune \
    --use-finetuned-clip \
    --clip-model-path YOUR-CKPT-PATH
```



### Training

Training TAG using pre-extracted feature (vanilla/fine-tuned CLIP):

```
python tag/train.py \
    --aokvqa-dir datasets/aokvqa \
    --vocab datasets/aokvqa/large_vocab_train.csv \
    --log-dir logs/ \
    --backbone clip \
    --clip-model-type ViT-L/14@336px \
    --train-features datasets/aokvqa/features/rationale_336px_train \
    --val-features datasets/aokvqa/features/rationale_336px_val \
    --vocab-features datasets/aokvqa/features/clip-ViT-L-14-336px-large-vocab.pt \
    --objective contrastive \
    --bart_path BART/bart-large \
    --inputs image \
    --name train_normal \
    --bs 32 
```



### Evaluation

```
python tag/eval_tag.py \
    --aokvqa-dir datasets/aokvqa \
    --vocab datasets/aokvqa/large_vocab_train.csv \
    --log-dir logs/ \
    --backbone clip \
    --clip-model-type ViT-L/14@336px \
    --train-features datasets/aokvqa/features/rationale_336px_train \
    --val-features datasets/aokvqa/features/rationale_336px_val \
    --vocab-features datasets/aokvqa/features/clip-ViT-L-14-336px-large-vocab.pt \
    --objective contrastive \
    --bart_path BART/bart-large \
    --inputs image \
    --ckpt YOUR-CKPT-PATH \
    --name train_normal \
    --bs 32 
```

