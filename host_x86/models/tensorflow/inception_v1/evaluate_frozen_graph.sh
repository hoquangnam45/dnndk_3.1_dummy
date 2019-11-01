#!/bin/sh

set -e

# Please set your imagenet validation dataset path here, 
IMAGE_DIR=/home/shengxiao/dataset/imagenet_image/val_resize_256/
IMAGE_LIST=/home/shengxiao/dataset/imagenet_image/val.txt

# Please set your batch size settings here, #IMAGES = VAL_BATCHES * BATCH_SIZE
# Commonly there are 5w image in total for imagenet validation dataset
EVAL_BATCHES=1000
BATCH_SIZE=50

python inception_v1_eval.py \
  --input_frozen_graph frozen_inception_v1.pb \
  --input_node input \
  --output_node InceptionV1/Logits/Predictions/Reshape_1 \
  --eval_batches $EVAL_BATCHES \
  --batch_size $BATCH_SIZE \
  --eval_image_dir $IMAGE_DIR \
  --eval_image_list $IMAGE_LIST \
  --gpu 0
