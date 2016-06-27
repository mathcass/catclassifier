"""Train a convolutional neural network to identify photos of my cat

"""

import os
import sys

import numpy as np
from skimage import io
from skimage import transform
from tflearn.data_utils import shuffle

from model import setup_model

SIZE = (32, 32)

image_dir = os.path.abspath("images")
cat = io.imread_collection(os.path.join(image_dir, "cat/*"))
not_cat = io.imread_collection(os.path.join(image_dir, "not_cat/*"))


X_cat = np.asarray([transform.resize(im, SIZE) for im in cat])
X_not_cat = np.asarray([transform.resize(im, SIZE) for im in not_cat])

X = np.concatenate((X_cat, X_not_cat), axis=0)
Y = np.concatenate((np.ones(X_cat.shape[0]),
                    np.zeros(X_not_cat.shape[0])))

Y = np.zeros((X.shape[0], 2))
Y[:X_cat.shape[0], 1] = np.ones(X_cat.shape[0])
Y[-X_not_cat.shape[0]:, 0] = np.ones(X_not_cat.shape[0])


n_training = int(X.shape[0] * .66)
X, Y = shuffle(X, Y)
X, X_test, Y, Y_test = X[:n_training], X[n_training:], Y[:n_training], Y[n_training:]

model = setup_model()

# Train it! We'll do 100 training passes and monitor it as it goes.
model.fit(X, Y, n_epoch=1000, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=100, snapshot_epoch=True,
          run_id='cat-classifier')
model.save("cat-classifier.tfl")
