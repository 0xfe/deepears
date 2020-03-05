# DeepEars

Mohit Muthanna Cheppudira <mohit@muthanna.com>

Various experiments in music analysis using deep learning.

## Generate Training Data

```
./gen_{chord|note}_samples
zip -r samples.zip samples/
gsutil -h "Cache-control:public,max-age=86400" -m cp -a public-read samples.zip gs://muthanna.com/deepears/instrument-samples-v1.zip
```

## Build and train

Use Jupyter to open one of the ipynb notebooks and follow instructions there.

