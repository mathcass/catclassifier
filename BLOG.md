# Using Deep Learning is Easier Than You Think

I came across a great
[article](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721#.469g9yw20)
on using the Deep Learning Python package `tflearn` to perform inference on some
classic datasets in Machine Learning like the MNIST dataset and the CIFAR-10
dataset.

Since I have a lot of experience working with data but not a lot working with
deep learning algorithms, I wondered how easy it would be to adapt these methods
to a new, somewhat related, problem. Turns out, it was easier than I thought.

As it turns out, these types of models have been around for quite a while for
various tasks in image recognition. The particular case of the
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) was solved by a
neural network very similar to the one from the mentioned post. The general idea
of using convolutional neural networks dates back to Yann LeCun's
[paper](https://www.cs.toronto.edu/~kriz/cifar.html) from 1998 in digit
recognition.

Training a similar neural network on my own data just amounted to connecting up
the inputs and the outputs properly. In particular, any input image has to be
re-scaled down (or up) to 32x32 pixels. Similarly, your output must be binary
and should represent membership of either of the two classes.

The main difficulty involves creating your dataset. This really just looks like
going through your images and classifying them by hand. For my own run at this,
all I did was create a directory like:

    images/
        cat/
        not_cat/

I put any cat photos I found into the `cat` directory while putting any non-cat
photographs in the other folder. I tried to keep the directories with about the
same number of images each to try to avoid any
[class imbalance problems](http://stats.stackexchange.com/questions/131255/class-imbalance-in-supervised-machine-learning).
Then again, this wasn't much of a concern since roughly half my photos are cat
photos anyway.

From there, `tflearn` has a helper method that lets you create a HDF5 dataset
from your directory of images with a
[simple function](http://tflearn.org/data_utils/#build-hdf5-image-dataset). The
`X` & `Y` values from that data structure can be used as the inputs to the deep
learning model.

By using around 400 images (roughly 200 for each class), my classifier achieved
about a 85% accuracy rate on a validation set of data. For my purposes, namely
just automatically tagging potential photos of my cat, this was accurate enough.
Any effort to increase the accuracy of this would probably involve any number
of:

* adding more training data by putting images into my class folders
* changing the shape of the network by adding more layers or more nodes per
  layer
* using a [pre-trained model](http://caffe.berkeleyvision.org/model_zoo.html) to
  bootstrap the learning process

That's all it really takes. If you know a bit of Python and can sort a few of
your photos into folders based on their category, you can get started using
sophisticated deep learning algorithms on your own images.

You can find the [code for this](https://github.com/mathcass/catclassifier) on
my account at Github. If you want to chat or reach out at all, follow me on
Twitter [@mathcass](http://twitter.com/mathcass).
