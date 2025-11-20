#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path /data2/gaodz/llava_v1.6_sft_stage2 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-v1.6-7b-stage2.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v2\


python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-v1.6-7b-stage2.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-v1.6-7b-stage2.json
