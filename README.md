## Quick Start (Post-installation)

Sets up python virtualenv, and AWS env variables.

    $ source env.sh
    $ mosh mohit@$TF1HOST (or $TFDEVHOST)
    tf$ . tfenv.sh

If instances aren't running:

    $ aws ec2 start-instances --instance-ids $TF1
    $ . env.sh

To stop:

    $ ssh ...; $ sudo shutdown now
    $ aws ec2 stop-instances --instance-ids $TF1

Sync:

    $ ./bootstrap/sync.sh

## MIDI Sample Generation

### Installation (Needs Ruby 2.0)

    $ gem install midilib
    $ brew install fluidsynth sox jq

### Get soundfont (see AWS setup instructions below)

    $ aws s3 cp s3://tftrain/soundfont.sf2 .

### Generate MIDI

  $ ./src/gen_midi_samples.rb
  $ play data/wav/*`

### To play

    $ fluidsynth -a coreaudio FluidR3_GM.sf2 "/Applications/Band-in-a-Box/Styles/Ear Training/Music Replay/MelodyReplay/MR02041.MID"

### To convert to WAV

    $  fluid synth -l -i -a file ~/w/audio/octave/FluidR3_GM.sf2 from_scratch.mid -F raw_audio
    $ sox -t raw -r 44100 -e signed -b 16 -c 2 raw_audio audio.wav

### Mix down to 1 channel and normalize

    $ sox -t raw -r 44100 -e signed -b 16 -c 2 test.raw ~/Downloads/test1.wav norm remix 2

(If you get clipping, try -b 32. If that doesn’t work add —norm)

### To create spectrogram

    $ sox ~/Downloads/test1.wav -n remix 2 spectrogram -x 400 -y 400 -m -r

(-m: monochrome, -r: raw, don’t show axis, remix: mix down two channels to 1)

Only first three seconds (trim):

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 3 spectrogram -x 400 -y 400 -m -o spectrogram.png

Quantize to 128 levels:

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 1 spectrogram -x 200 -y 200 -m -r -q 128 -o spectrogram.png

To add noise:

    $ sox ~/Downloads/test1.wav -p synth whitenoise vol 0.1 | sox -m ~/Downloads/test1.wav - ~/Downloads/noisy.wav

## TensorFlow Audio Training (Mac)

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

Download cuDNN for cuda version OSX from: https://developer.nvidia.com/rdp/cudnn-download



### AWS

Install AWS CLI.

    $ pip install awscli
    $ brew install jq

Decrypt keys in `awskey.csv.gpg` and SSH private key in `tftrain.pem`. Then configure AWS.

    $ gpg2 -d awskey.csv.gpg (access key and secret key)
    $ gpg2 -d tftrain.pem.gpg
    $ chmod 400 tftrain.pem awskey.csv
    $ aws configure (region: us-east-1) 

### AWS Tools

Create AWS GPU instance using AMI: `ami-fce3c696` (Ubuntu Trusty 14.04)

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type g2.2xlarge --key-name tftrain --security-groups dev --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 32 } } ]"

Create non-GPU dev instance (`t2.micro`, `m4.large`) for development:

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type t2.micro --key-name tftrain --subnet-id subnet-91734fac --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 32 } } ]"

Add a name tag:

    $ aws ec2 describe-instances
    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tftrain1 Key=project,Value=deepears # GPU
    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tfdev Key=project,Value=deepears    # CPU

Get Instance ID and public DNS name:

    $ aws ec2 describe-instances --filters Name=tag:id,Values=tftrain1
    $ aws ec2 describe-instances --filters Name=tag:id,Values=tftrain1 --query 'Reservations[0].Instances[0].InstanceId'
    $ aws ec2 describe-instances --filters Name=tag:id,Values=tftrain1 --query 'Reservations[0].Instances[0].PublicDnsName'
    $ TF1=`aws ec2 describe-instances --filters Name=tag:id,Values=tftrain1 | jq -r '.Reservations[0].Instances[0].InstanceId'`
    $ TF1HOST=`aws ec2 describe-instances --filters Name=tag:id,Values=tftrain1 | jq -r '.Reservations[0].Instances[0].PublicDnsName'`

SSH to machine:

    $ ssh -i tftrain.pem  ubuntu@ec2-54-242-223-138.compute-1.amazonaws.com

Add the following to ~/.ssh/config (so you don't have to keep typing -i):

    Host *.compute-1.amazonaws.com
        IdentityFile ~/w/audio/train/tftrain.pem

If you have ssh-agent: `$ ssh-add ~/.ssh/KEY_PAIR_NAME.pem`

## Setup AWS GPU instance or GCE highcpu instance

### AWS

Instance prices: https://aws.amazon.com/ec2/pricing/. Use `g2.2xlarge` for GPU (with 8 CPU cores, 16GB RAM), and `c4.4xlarge` for 16-core CPU (30GB RAM).

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type g2.2xlarge --key-name tftrain --security-groups dev --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 64 } } ]"

    (Check aws.amazon.com to verify that instance is running)

    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tftrain1
    $ . env.sh 

### GCE

    $ gcloud compute instances create "tftrain" --machine-type "n1-highcpu-16" --network "default" --metadata "id=gtftrain1" --no-restart-on-failure --maintenance-policy "MIGRATE" --scopes default="https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management" --tags "http-server","https-server","deepears" --image "/ubuntu-os-cloud/ubuntu-1404-trusty-v20160627" --boot-disk-size "80" --boot-disk-type "pd-standard" --boot-disk-device-name "tftrain"

    # create keys and update ~/.ssh/config
    $ gcloud compute config-ssh
    $ . env.sh

### Install DeepEars

Seed server:

    $ cd ~/w/static
    $ AWS: ./seed.sh $TF1HOST tftrain1 ubuntu
    $ GCE: ./seed.sh $TF1GHOST tftrain1 mmuthanna
    $ ssh mohit@$TF1HOST
    aws$ passwd
    aws$ cd bootstrap
    aws$ ./bootstrap-user.sh
    aws$ ./install-tools.sh
    aws$ sudo vi /etc/hosts (add tftrain1 to localhost)

Seed deepears:

    $ aws s3 cp --recursive s3://tftrain/ bootstrap/assets/ --include '*'
    $ ./bootstrap/seed.sh $TF1HOST
    $ ssh mohit@$TF1HOST
    $ screen
    $ cd bootstrap; ./install-gpu.sh

Reboot machine and test nvidia:

    $ sudo shutdown -r now
    (... wait to ssh back in ...)
    $ nvidia-smi
    Mon Jun 27 11:25:34 2016       
    +------------------------------------------------------+                       
    | NVIDIA-SMI 352.93     Driver Version: 352.93         |                       
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GRID K520           Off  | 0000:00:03.0     Off |                  N/A |
    | N/A   32C    P0    35W / 125W |     11MiB /  4095MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

Test:

    $ ssh mohit@$TF1HOST
    aws$ . tfenv.sh
    aws$ python
    >>> import tensorflow as tf
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.so locally
    I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.so locally
    >>> a = tf.constant("Boo!")
    >>> tf.Session().run(a)

Build from source:

    $ cd bootstrap; ./build-tensorflow.sh

## Google Cloud

Docs: https://cloud.google.com/compute/docs/gcloud-compute/

    $ gcloud config configurations activate deepears
    $ gcloud auth login
    $ gcloud config list
    $ gcloud compute instances list

Create big machine:

    $ gcloud compute instances create "tftrain" --machine-type "n1-highcpu-16" --network "default" --metadata "id=gtftrain1" --no-restart-on-failure --maintenance-policy "MIGRATE" --scopes default="https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management" --tags "http-server","https-server","deepears" --image "/ubuntu-os-cloud/ubuntu-1404-trusty-v20160627" --boot-disk-size "80" --boot-disk-type "pd-standard" --boot-disk-device-name "tftrain"
    $ gcloud compute instances stop tftrain
    $ gcloud compute instances start tftrain

    # create keys and update ~/.ssh/config
    $ gcloud compute config-ssh

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

### Training

First convert WAV files to TFRecord format:

    $ src/tfbuild.py

Start training:

    $ src/tftrain.py  (OR)
    $ src/tftrain_deep.py
 
 Start tensorboard:

    $ $ tensorboard --logdir model --port 3000

## Log (Singles)

### June 22 2016
Input layer: 25800
Output layer: 12 (softmax)
No hidden layers (simple linear model with softmax output   )
Train batch size: 10
Test batch size: 100
Number of examples/batches: ~19000, 1900 batches

- LR=0.5, Accuracy: .45 (peak, after 1800 batches), no confusion matrix
- LR=10, Accuracy: .52 (hover, after about 500 batches), cm-02.png
   + CM looks pretty good
   + lots of G predictions

### June 27 2016
DNNClassifier (tftrain_deep.py)
Layers: 25800, 1000, 100, 12
LR: 0.1
STEPS: 10000
Dropout: 0.1
Test/training accuracy: 0.73/0.83
Test/training loss: 0.64/0.42

### June 28 2016
Same as above.
STEPS: 20000
LR: 0.1
Test: {'accuracy': 0.74781251, 'loss': 0.67122149}  

Layers: 25800, 1000, 300, 12
STEPS: 10000
LR: 0.3 (fail)
LR: 0.1  
Test: {'accuracy': 0.63575, 'loss': 0.95323491}

Layers: 25800, 1000, 300, 50, 12
STEPS: 12000
LR: 0.2  
Dropout: 0.2
Test: loss = 0.961167, accuracy = 0.634277

LR: 0.2
16:03:38.733830 (training): evaluation (step 18700): loss = 1.08154, accuracy = 0.599609  

CPU (GCE):
19:32:39.324272 (validation): evaluation (step 19900): loss = 0.625723, accuracy = 0.757324
19:32:43.120241 (training): evaluation (step 19900): loss = 0.286668, accuracy = 0.883301