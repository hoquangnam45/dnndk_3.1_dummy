#!/usr/bin/env bash

#working directory
work_dir=$(pwd)
#path of float model
model_dir=${work_dir}
#output directory
output_dir=${work_dir}/decent_output

if [ -f /usr/local/bin/decent ]; then
    DECENT="decent"
elif [ -f /usr/local/bin/decent-cpu ]; then
    DECENT="decent-cpu"
else
    echo "Error: Please run DNNDK host_x86/install.sh first to install decent"
    exit 1
fi

[ -d "$output_dir" ] || mkdir "$output_dir"

$DECENT    quantize                               \
           -model ${model_dir}/float.prototxt     \
           -weights ${model_dir}/float.caffemodel \
           -output_dir ${output_dir}              \
           -method 1
