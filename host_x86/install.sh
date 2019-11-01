#!/bin/bash

board_name=$1
DNNDK_VERSION="DNNDK_VERSION=3.1"
support_board=(ZCU102 ZCU104 ZedBoard Ultra96)
dpu_versions=(1.4.0 1.4.0 1.4.0 1.4.0)

install_fail () {
    echo "Installation failed!" 1>&2
    exit 1
}
print_help() {
    echo "Usage: install.sh board_name"
    echo "The board_name is:"
    for (( i=0; i<$(echo ${#support_board[@]}); i++ )); do
        echo "    ${support_board[$i]}"
    done
}

# Support for only one dnnc version
one_dnnc=""
if [ $(ls -l pkgs/ubuntu16.04/dnnc-*|grep ^-|wc -l) = 1 ]; then
    one_dnnc=true
fi
# Support for multi dnnc versions
if [ "$one_dnnc" != "true" ]; then
    if [ "$board_name" = "" ]; then
        print_help
        install_fail
    fi

    verify=""
    for (( i=0; i<$(echo ${#support_board[@]}); i++ )); do
        if [ "$board_name" = "${support_board[$i]}" ]; then
            verify=true;
        fi
    done

    if [ "$verify" != "true" ]; then
        echo "Unsupported board_name $board_name"
        print_help
        install_fail
    fi
fi

echo "Inspect system environment ..."

echo "[system version]"
sysver=`lsb_release -a | grep Description:`
#echo $sysver
sysver=`echo $sysver | awk '{print $3}' | awk -F'.' '{print $1"."$2}' `
echo $sysver

cuda_verfile="/usr/local/cuda/version.txt"
cudnn_verfile="/usr/local/cuda/include/cudnn.h"
cuda_ver=
cudnn_ver=

if [ -f $cuda_verfile ] ; then
    cuda_ver=`cat $cuda_verfile | awk '{ print $3 } ' `
    cuda_ver=`echo $cuda_ver | awk -F'.' '{print $1"."$2}'`
fi

if [ -f $cudnn_verfile ] ; then
    cudnn_ver=`cat $cudnn_verfile | grep CUDNN_MAJOR -A 2 | head -3 | awk '{ver=ver$3"."}END{print ver}'`
    cudnn_ver=`echo $cudnn_ver | awk -F'.' '{print $1"."$2"."$3}'`
fi

if [ "$cuda_ver" != "" ]; then
    echo "[CUDA version]"
    echo $cuda_ver
fi

if [ "$cudnn_ver" != "" ]; then
    echo "[CUDNN version]"
    echo $cudnn_ver
fi

echo  "Begin to install Xilinx DNNDK tools on host ..."
msg=$(rm /usr/local/bin/dnnc* 2>&1)
msg=$(rm /usr/local/bin/decent* 2>&1)
msg=$(rm /usr/local/bin/dlet 2>&1)
msg=$(rm /usr/local/bin/ddump 2>&1)

# install dnnc, and CPU version of decent caffe
array=(
        ubuntu14.04
        ubuntu16.04
        ubuntu18.04
    )
for data in ${array[@]}; do
    if [[ $data =~ $sysver ]]; then
        tmp_dir="$(pwd)"
        cd pkgs/${data}
        cp dnnc-* /usr/local/bin || install_fail
        cp decent-cpu /usr/local/bin || install_fail
        cp dlet /usr/local/bin || install_fail
        cp ddump /usr/local/bin || install_fail
        chmod 755 /usr/local/bin/dnnc* || install_fail
        chmod 755 /usr/local/bin/decent-cpu || install_fail
        chmod 755 /usr/local/bin/dlet || install_fail
        chmod 755 /usr/local/bin/ddump || install_fail
        cd "$tmp_dir"
    fi
done

# set dnnc link
if [ "$one_dnnc" = "true" ]; then
    conf_dpu_version=${dpu_versions[0]}
    tmp=${support_board[@]}
    conf_board_name=${tmp// /,}
    backup_dir=$(pwd)
    cd /usr/local/bin
    ln -s "$(ls dnnc-*)" /usr/local/bin/dnnc || install_fail
    cd "$backup_dir"
else
    for (( i=0; i<${#support_board[@]}; i++ )); do
        if [ "$board_name" = "${support_board[$i]}" ]; then
            conf_dpu_version=${dpu_versions[$i]}
            conf_board_name=${support_board[$i]}
            ln -s "dnnc-dpu${dpu_versions[$i]}" /usr/local/bin/dnnc || install_fail
        fi
    done
fi

echo "# This is the configuration file for Xilinx DNNDK." > /etc/dnndk.conf || exit 1
echo "# It is automatically generated by host_x86/install.sh of DNNDK package." >> /etc/dnndk.conf  || exit 1
echo "" >> /etc/dnndk.conf  || exit 1
echo "$DNNDK_VERSION" >> /etc/dnndk.conf  || exit 1
echo "DPU_VERSION=$conf_dpu_version" >> /etc/dnndk.conf  || exit 1
echo "BOARD_NAME=$conf_board_name" >> /etc/dnndk.conf  || exit 1
echo "" >> /etc/dnndk.conf  || exit 1
chmod 644 /etc/dnndk.conf || exit 1
echo "Complete dnnc installation successfully."
echo "Complete CPU version of decent for caffe installation successfully."

# install GPU version of decent for caffe
array=(
    ubuntu14.04/cuda_8.0.61_GA2_cudnn_v7.0.5
    ubuntu16.04/cuda_8.0.61_GA2_cudnn_v7.0.5
    ubuntu16.04/cuda_9.0_cudnn_v7.0.5
    ubuntu16.04/cuda_9.1_cudnn_v7.0.5
    ubuntu18.04/cuda_10.0_cudnn_v7.4.1
)

for data in ${array[@]}; do
    if [[ "$cuda_ver" = "" ]] || [[ "$cudnn_ver" = "" ]]; then
        break
    fi
    if [[ $data =~ $sysver ]] && [[ $data =~ $cuda_ver ]] && [[ $data =~ $cudnn_ver ]] ; then
        tmp_dir="$(pwd)"
        cd pkgs/${data} || install_fail
        cp decent /usr/local/bin || install_fail
        chmod 755 /usr/local/bin/decent || install_fail
        cd "$tmp_dir"

        echo "Complete GPU version of decent installation successfully."
        exit 0
    fi
done

if [ "$decent_cuda_installed" != true ]; then
    echo "The host system environment supported by GPU version of decent for caffe is as follows:"
    echo "1 - Ubuntu 14.04 + CUDA 8.0 + cuDNN 7.05"
    echo "2 - Ubuntu 16.04 + CUDA 8.0 + cuDNN 7.05"
    echo "3 - Ubuntu 16.04 + CUDA 9.0 + cuDNN 7.05"
    echo "4 - Ubuntu 16.04 + CUDA 9.1 + cuDNN 7.05"
    echo "5 - Ubuntu 18.04 + CUDA 10.0 + cuDNN 7.41"
fi
echo "But does not meet the above environment."
echo "The GPU version of decent for caffe installation failed."
exit 1
