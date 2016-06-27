### Quick Start (Post-installation)

Sets up python virtualenv, and AWS env variables.

    $ source env.sh
    $ ssh mohit@$TF1HOST (or $TFDEVHOST)

### MIDI Sample Generation

## Installation

    $ gem install midilib
    $ brew install fluidsynth sox jq

## Get soundfont

    Browse: https://musescore.org/en/handbook/soundfont#list

## Generate MIDI

  $ ./gen_midi_samples.rb
  $ play data/wav/*`

## To play

    $ fluidsynth -a coreaudio FluidR3_GM.sf2 "/Applications/Band-in-a-Box/Styles/Ear Training/Music Replay/MelodyReplay/MR02041.MID"

## To convert to WAV

    $  fluid synth -l -i -a file ~/w/audio/octave/FluidR3_GM.sf2 from_scratch.mid -F raw_audio
    $ sox -t raw -r 44100 -e signed -b 16 -c 2 raw_audio audio.wav

## Mix down to 1 channel and normalize

    $ sox -t raw -r 44100 -e signed -b 16 -c 2 test.raw ~/Downloads/test1.wav norm remix 2

(If you get clipping, try -b 32. If that doesn’t work add —norm)

## To create spectrogram

    $ sox ~/Downloads/test1.wav -n remix 2 spectrogram -x 400 -y 400 -m -r

(-m: monochrome, -r: raw, don’t show axis, remix: mix down two channels to 1)

Only first three seconds (trim):

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 3 spectrogram -x 400 -y 400 -m -o spectrogram.png

Quantize to 128 levels:

    $ sox ~/Downloads/test1.wav -n remix 2 trim 0 1 spectrogram -x 200 -y 200 -m -r -q 128 -o spectrogram.png

To add noise:

    $ sox ~/Downloads/test1.wav -p synth whitenoise vol 0.1 | sox -m ~/Downloads/test1.wav - ~/Downloads/noisy.wav

### TensorFlow Audio Training

## Install NumPy, SciPy, Pandas, TensorFlow on Mac

    $ sudo easy_install pip
    $ sudo pip install --upgrade virtualenv
    $ virtualenv tf
    $ source tf/bin/activate

Install numerical modules:

    (tf)$ pip install scipy pandas matplotlib sklearn numpy seaborn tqdm

Install Tensorflow:

    $ export TF_BIN=<get url from https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#virtualenv-installation>
    (tf)$ pip install --upgrade $TF_BIN
    
Note: You may need to add flag to 'pip install' of TF: --ignore-installed six (details)

## AWS

Install AWS CLI.

    $ pip install awscli

Decrypt keys in `awskey.csv.gpg` and SSH private key in `tftrain.pem`. Then configure AWS.

    $ gpg2 -d awskey.csv.gpg (access key and secret key)
    $ gpg2 -d tftrain.pem.gpg
    $ chmod 400 tftrain.pem awskey.csv
    $ aws configure (region: us-east-1) 

## AWS Tools

Create AWS GPU instance using AMI: `ami-fce3c696` (Ubuntu Trusty 14.04)

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type g2.2xlarge --key-name tftrain --security-groups dev --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 32 } } ]"

Create non-GPU dev instance (`t2.micro`, `m4.large`) for development:

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type t2.micro --key-name tftrain --subnet-id subnet-91734fac --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 32 } } ]"

Add a name tag:

    $ aws ec2 describe-instances
    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tftrain1  # GPU
    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tfdev     # CPU

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

Seed server:

    $ cd ~/w/static
    $ ./seed.sh aws.dns.address hostname ubuntu
    $ ssh mohit@aws.dns
    aws$ passwd
    aws$ cd bootstrap
    aws$ ./bootstrap-user.sh
    aws$ ./install-gpu.sh

Copy CuDNN:

    $ scp cudnn-7.0-linux-x64-v4.0-prod.tgz mohit@ec2-52-87-235-160.compute-1.amazonaws.com:tmp
    $ ssh mohit@aws.dns
    aws$ sudo cp cuda/include/* /usr/local/cuda/include
    aws$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
    aws$ sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

Add to `.bashrc` (probably already there):

    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
    export CUDA_HOME=/usr/local/cuda

## Training

First convert WAV files to TFRecord format:

    $ ./tfbuild.py

Start training:

    $ ./tftrain.py

## Log (Singles)

# June 22 2016
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

