# Setup environment variables for TensorFlow:
#
# To run: $ source env.sh

TFHOME=$HOME/tf
. $TFHOME/bin/activate

# GPU stuff
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
export CUDA_HOME=/usr/local/cuda
