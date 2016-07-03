# Cat Classifier

I think deep learning is accessible enough now that if you know how to program,
you know how to get started using it for your own tasks. This great
[article](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.469g9yw20)
shows you how you can use `tflearn`, a [TensorFlow](https://www.tensorflow.org/)
based Python library to create predictive models based on the CIFAR-10 dataset.

This is a simple implementation using the same neural network layout to identify
labeled photos of my cat.

## Run

To run this, first create an Anaconda environment based off the
`environment.yml` using Python 3.5. Then, create a folder `images` in the local
directory with two subfolders `cat` and `not_cat`. Sort through your own files
and copy your cat photos into `cat` and your non-cat photos into `not_cat`.

To run the training step, run:

    python cnn.py

which will read all of the files and train a network based on the image
features. That script will also write to a file `cat-classifier.tfl` which is a
binary representation of the trained model that you can use in later scripts.

To use a trained model from `cnn.py` to classify your own images, run:

    python classify.py <image_path>

where `<image_path>` is the path of an image you want to classify. The output
from this is a JSON object with probabilities for `cat` and `not_cat` scores.
