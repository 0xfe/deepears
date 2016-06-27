# To run: $ source env.sh

TFHOME=./tf
. $TFHOME/bin/activate

# GPU stuff
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda

CPUHOST=tfdev
GPUHOST=tftrain1

# Hosts
TFDEV=`aws ec2 describe-instances --filters Name=tag:id,Values=$CPUHOST | jq -r '.Reservations[0].Instances[0].InstanceId'`
TFDEVHOST=`aws ec2 describe-instances --filters Name=tag:id,Values=$CPUHOST | jq -r '.Reservations[0].Instances[0].PublicDnsName'`

TF1=`aws ec2 describe-instances --filters Name=tag:id,Values=$GPUHOST | jq -r '.Reservations[0].Instances[0].InstanceId'`
TF1HOST=`aws ec2 describe-instances --filters Name=tag:id,Values=$GPUHOST | jq -r '.Reservations[0].Instances[0].PublicDnsName'`

echo "CPU ($CPUHOST):"
echo "  TFDEV=$TFDEV"
echo "  TFDEVHOST=$TFDEVHOST"
echo "GPU ($GPUHOST):"
echo "  TF1=$TF1"
echo "  TF1HOST=$TF1HOST"
