#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /data2/gaodz/llava_v1.6_sft_full \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.6-7b-sft.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v2

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llava-v1.6-7b-sft.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llava-v1.6-7b-sft.json

