#!/bin/bash
# Installs deep learning and audio processing packages for GPU servers:
#
#   - sox, fluidsynth, fluidsynth soundbank
#   - ruby gem: midilib
#   - numpy, scipy, CUDA, tensorflow
#
# Prerequisites: bootstrap.sh run as root.

THISDIR=`pwd`
mkdir -p ~/tmp

CUDA_URL=http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl

cp tfenv.sh $HOME
cd $HOME

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install libprotobuf-dev libleveldb-dev\
     libatlas-base-dev fluid-soundfont-gm fluidsynth sox gfortran\
     python-setuptools python-setuptools-git python-pip\
     python-dev libpng12-dev libfreetype6-dev libxft-dev libncurses-dev\
     libopenblas-dev gfortran libblas-dev liblapack-dev libatlas-base-dev\
     python-pydot linux-headers-generic linux-image-extra-virtual unzip\
     swig unzip wget pkg-config zip g++ zlib1g-dev

sudo gem install midilib

# Tensorflow and math stuff
sudo pip install --upgrade virtualenv
virtualenv tf
source tf/bin/activate
pip install scipy pandas matplotlib sklearn numpy seaborn tqdm

# CUDA and cuDNN (for GPU)
# https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#optional-install-cuda-gpus-on-linux

cd ~/tmp
curl -O $CUDA_URL
sudo dpkg -i cuda-*.deb
sudo apt-get update
sudo apt-get install cuda

cd ~/tmp
tar -zxvf $THISDIR/bootstrap/assets/cudnn-7.0-linux-x64-v4.0-prod.tgz
sudo cp cuda/include/* /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

pip install --upgrade $TF_BINARY_URL