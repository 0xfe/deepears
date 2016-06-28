# To run: $ source env.sh

TFHOME=./tf
. $TFHOME/bin/activate

# GPU stuff
export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64"

CPUHOST=tfdev
GPUHOST=tftrain1

# Hosts
export TFDEV=`aws ec2 describe-instances --filters Name=tag:id,Values=$CPUHOST | jq -r '.Reservations[0].Instances[0].InstanceId'`
export TFDEVHOST=`aws ec2 describe-instances --filters Name=tag:id,Values=$CPUHOST | jq -r '.Reservations[0].Instances[0].PublicDnsName'`

export TF1=`aws ec2 describe-instances --filters Name=tag:id,Values=$GPUHOST | jq -r '.Reservations[0].Instances[0].InstanceId'`
export TF1HOST=`aws ec2 describe-instances --filters Name=tag:id,Values=$GPUHOST | jq -r '.Reservations[0].Instances[0].PublicDnsName'`

echo "CPU ($CPUHOST):"
echo "  TFDEV=$TFDEV"
echo "  TFDEVHOST=$TFDEVHOST"
echo "GPU ($GPUHOST):"
echo "  TF1=$TF1"
echo "  TF1HOST=$TF1HOST"
