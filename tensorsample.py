import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print(tf.__version__)
print(tf.keras.__version__)