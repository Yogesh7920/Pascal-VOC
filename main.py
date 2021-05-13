import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
import sys
from config import Config
import numpy as np
import cv2


@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.

    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)

    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
    cost = 1 - soft_f1
    macro_cost = tf.reduce_mean(cost)
    return macro_cost


if __name__ == '__main__':

    labels = Config.labels[:-1]
    labels[-1] = 'tv/moniter'
    labels = np.array(labels)

    args = sys.argv
    args.pop(0)

    img = plt.imread(args[1])
    img = img[:, :, :3]
    if np.max(img) == 1:
        img *= 255
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    img = preprocess_input(img)
    img = img.reshape(1, 224, 224, 3)

    if args[0] == '1':
        model = keras.models.load_model(f'models/task{args[0]}.h5', compile=False)
        preds = (model.predict(img) > 0.5) * 1
        print(labels[np.where(preds[0] == 1)])

    elif args[0] == '2':
        model = keras.models.load_model(f'models/model2', compile=False)
        with open(args[2]) as f:
            annots = f.read()

        preds = (model.predict((img, [annots])) > 0.5) * 1
        print(labels[np.where(preds[0] == 1)])

    elif args[0] == '3':
        pass
    else:
        print("Wrong argument")
