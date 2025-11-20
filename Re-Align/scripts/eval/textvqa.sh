#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /data2/gaodz/llava-v1.6-vicuna-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.6-7b-tga_subnet.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1\
    --load-peft /data2/gaodz/tga_subnet_amss
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.6-7b-tga_subnet.jsonl
