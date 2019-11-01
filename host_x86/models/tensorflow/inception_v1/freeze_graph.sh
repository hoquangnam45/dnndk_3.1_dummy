#!/bin/sh
set -e

freeze_graph \
  --input_graph=./float_graph/inception_v1_inf_graph.pb \
  --input_checkpoint=./float_graph/inception_v1.ckpt \
  --input_binary=true \
  --output_graph=./frozen_inception_v1.pb \
  --output_node_names=InceptionV1/Logits/Predictions/Reshape_1
