#!/bin/bash

python main.py \
    --mode eval \
    --gpu 0 \
    --exper-name test_eval \
    --eval-checkpoint /content/drive/MyDrive/Graph_Classroom/RAER-Education/CLIP-CAER/outputs/test-[12-10]-[08:55]/model_best.pth \
    --root-dir /content/drive/MyDrive/Graph_Classroom/RAER-Education/RAER \
    --test-annotation /content/drive/MyDrive/Graph_Classroom/RAER-Education/RAER/annotation/test.txt \
    --clip-path ViT-B/32 \
    --bounding-box-face /content/drive/MyDrive/Graph_Classroom/RAER-Education/RAER/bounding_box/face.json \
    --bounding-box-body /content/drive/MyDrive/Graph_Classroom/RAER-Education/RAER/bounding_box/body.json \
    --graph_dir /content/drive/MyDrive/Graph_Classroom/RAER-Education/RAER/graph \
    --text-type class_descriptor \
    --contexts-number 8 \
    --class-token-position end \
    --class-specific-contexts True \
    --load_and_tune_prompt_learner True \
    --temporal-layers 1 \
    --num-segments 16 \
    --duration 1 \
    --image-size 224 \
    --seed 42
