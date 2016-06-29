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
rm ~/tmp/*
. ~/tfenv.sh
export DEBIAN_FRONTEND=noninteractive

BAZEL_BUILD_GPU="--config=cuda"
BAZEL_BUILD_CPU=""
BAZEL_BUILD_FLAGS=$BAZEL_BUILD_GPU

# Install bazel: http://www.bazel.io/docs/install.html#install-on-ubuntu

sudo add-apt-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get install oracle-java8-installer

echo "deb http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install bazel

update-alternatives --set java /usr/lib/jvm/java-8-oracle/jre/bin/java
export JAVA_HOME=/usr/lib/jvm/java-8-oracle

mkdir -p ~/git
cd ~/git
git clone https://github.com/tensorflow/tensorflow
cd tensorflow

echo Running configure script... use defaults for everything except
echo for Cuda compute capabilities. AWS supports only 3.0.
echo ""
echo Hit return to start.
read

./configure
bazel build -c opt $BAZEL_BUILD_FLAGS //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tmp/tensorflow_pkg
pip install --upgrade ~/tmp/tensorflow_pkg/*.whl