# Cloud Setup Instructions

## AWS Environment Setup

Install AWS CLI.

    $ pip install awscli
    $ brew install jq

Decrypt keys in `awskey.csv.gpg` and SSH private key in `tftrain.pem`. Then configure AWS.

    $ gpg2 -d awskey.csv.gpg (access key and secret key)
    $ gpg2 -d tftrain.pem.gpg
    $ chmod 400 tftrain.pem awskey.csv
    $ aws configure (region: us-east-1) 

## Google Cloud (GCE) Environment Setup

Docs: https://cloud.google.com/compute/docs/gcloud-compute/

    $ gcloud config configurations activate deepears
    $ gcloud auth login
    $ gcloud config list
    $ gcloud compute instances list

## Setup AWS GPU instance or GCE highcpu instance

### AWS

Instance prices: https://aws.amazon.com/ec2/pricing/. Use `g2.2xlarge` for GPU (with 8 CPU cores, 16GB RAM), and `c4.4xlarge` for 16-core CPU (30GB RAM).

    $ aws ec2 run-instances --image-id ami-fce3c696 --count 1 --instance-type g2.2xlarge --key-name tftrain --security-groups dev --block-device-mapping "[ { \"DeviceName\": \"/dev/sda1\", \"Ebs\": { \"VolumeSize\": 64 } } ]"

    (Check aws.amazon.com to verify that instance is running)

    $ aws ec2 create-tags --resources i-xxxxxxxx --tags Key=id,Value=tftrain1
    $ . env.sh 

### GCE

    $ gcloud compute instances create "tftrain" --machine-type "n1-highcpu-16" --network "default" --metadata "id=gtftrain1" --no-restart-on-failure --maintenance-policy "MIGRATE" --scopes default="https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management" --tags "http-server","https-server","deepears" --image "/ubuntu-os-cloud/ubuntu-1404-trusty-v20160627" --boot-disk-size "80" --boot-disk-type "pd-standard" --boot-disk-device-name "tftrain"
    $ gcloud compute config-ssh
    $ . env.sh

### Install DeepEars (GCE or AWS)

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

## Random Cloud Notes

### GCE

Create big machine:

    $ gcloud compute instances create "tftrain" --machine-type "n1-highcpu-16" --network "default" --metadata "id=gtftrain1" --no-restart-on-failure --maintenance-policy "MIGRATE" --scopes default="https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring.write","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management" --tags "http-server","https-server","deepears" --image "/ubuntu-os-cloud/ubuntu-1404-trusty-v20160627" --boot-disk-size "80" --boot-disk-type "pd-standard" --boot-disk-device-name "tftrain"
    $ gcloud compute instances stop tftrain
    $ gcloud compute instances start tftrain

    # create keys and update ~/.ssh/config
    $ gcloud compute config-ssh

### AWS

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