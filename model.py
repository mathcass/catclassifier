"""Sets up model code for training on cat dataaset
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# Define our network architecture:

def setup_model(checkpoint_path=None):
    """Sets up a deep belief network for image classification based on the set up described in

    :param checkpoint_path: string path describing prefix for model checkpoints
    :returns: Deep Neural Network
    :rtype: tflearn.DNN

    References:
        - Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks

    Links:
        - https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721

    """
     # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)

    # Input is a 32x32 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    if checkpoint_path:
        model = tflearn.DNN(network, tensorboard_verbose=3,
                            checkpoint_path=checkpoint_path)
    else:
        model = tflearn.DNN(network, tensorboard_verbose=3)

    return model

