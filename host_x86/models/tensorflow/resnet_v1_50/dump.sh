decent_q dump \
  --input_frozen_graph quantize_results/quantize_eval_model.pb \
  --input_fn resnet_v1_50_input_fn.calib_input \
  --max_dump_batches 2 \
  --dump_float 0 \
  --output_dir ./quantize_results \
