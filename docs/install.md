## Setup TensorFlow and other Audio Libraries on Mac

### MIDI and Audio Package Installation

    $ gem install midilib (needs ruby2.0)
    $ brew install fluidsynth sox jq

### Install NumPy, SciPy, Pandas, TensorFlow on Mac

    $ sudo easy_install pip
    $ sudo pip install --upgrade virtualenv
    $ virtualenv tf
    $ source tf/bin/activate

Install numerical modules:

    (tf)$ pip install scipy pandas matplotlib sklearn numpy seaborn tqdm awscli

Install Tensorflow:

    $ export TF_BIN=<get url from https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation>
    (tf)$ pip install --upgrade $TF_BIN
    
Note: You may need to add flag to 'pip install' of TF: --ignore-installed six (details)

### Install Tensorflow from Source

    $ brew install bazel swig coreutils
    $ brew cask install cuda (note down version)

Download cuDNN for cuda version OSX from: https://developer.nvidia.com/rdp/cudnn-downloadA

## CuDNN Setup

Copy CuDNN:

    $ aws s3 cp cudnn-7.0-linux-x64-v4.0-prod.tgz .
    $ scp cudnn-7.0-linux-x64-v4.0-prod.tgz mohit@ec2-52-87-235-160.compute-1.amazonaws.com:tmp
    $ ssh mohit@aws.dns
    aws$ sudo cp cuda/include/* /usr/local/cuda/include
    aws$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    aws$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

Add to `.bashrc` (probably already there):

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
    export CUDA_HOME=/usr/local/cuda