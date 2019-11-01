#!/bin/sh
set -e

freeze_graph \
  --input_graph=./float_graph/resnet_v1_50_inf_graph.pb \
  --input_checkpoint=./float_graph/resnet_v1_50.ckpt \
  --input_binary=true \
  --output_graph=./frozen_resnet_v1_50.pb \
  --output_node_names=resnet_v1_50/predictions/Reshape_1
