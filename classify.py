"""Classifies image arguments as either cat or not

"""

import json
import sys
import os

from skimage import io
from skimage import transform

from model import setup_model

# TODO: handle argument parsing in a sane manner

SIZE = (32, 32)
MODEL_FILE = "cat-classifier.tfl"

model = setup_model()

model.load(MODEL_FILE)

filenames = sys.argv[1:]

for filename in filenames:
    filepath = os.path.abspath(filename)
    try:
        im = io.imread(filepath)
        im = transform.resize(im, SIZE)
    except ValueError:
        print("Unable to load: {:s}".format(filepath), file=sys.stderr)
    result = model.predict([im])
    print(json.dumps({
        "filepath": filepath,
        "not_cat_score": result[0][0],
        "cat_score": result[0][1]
    }))
