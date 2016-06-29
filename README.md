## Quick Start (Post-installation)

Sets up python virtualenv, and AWS/GCE env variables:

    $ source env.sh
    $ mosh mohit@$TF1HOST (or $TF1GHOST for GCE)
    tf$ . tfenv.sh

If instances aren't running:

    $ aws ec2 start-instances --instance-ids $TF1
    $ . env.sh

    $ gcloud compute instances start tftrain

To stop:

    $ ssh ...; $ sudo shutdown now
    $ aws ec2 stop-instances --instance-ids $TF1
    $ gcloud compute instances stop tftrain

Sync (update host in `bootstrap/sync.sh` to AWS or GCE):

    $ ./bootstrap/sync.sh

### Training

To build dataset:

    $ src/gen.sh
    $ src/tfbuild.py
    (optional)$ aws s3 cp data/...tfrecords.tar.gz s3://tftrain/...

Start training:

    $ src/tftrain.py  (OR)
    $ src/tftrain_deep.py
 
Start tensorboard:

    $ tensorboard --logdir model --port 3000

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