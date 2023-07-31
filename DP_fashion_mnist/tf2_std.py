import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# fashion_mnist图像分类数据集   60000个样本，每个样本28*28的特征  target 0-9 对应不同衣服种类
fashion_mnist = keras.datasets.fashion_mnist
# 数据加载本地用户.keras文件中
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
# 验证集、训练集
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

std = StandardScaler()
x_train_std = std.fit_transform(x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_std = std.transform(x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_test_std = std.transform(x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(0.001),
              metrics=["accuracy"])

history = model.fit(x_train_std, y_train, epochs=20,
                    validation_data=(x_valid_std, y_valid))


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

model.evaluate(x_test_std, y_test, verbose=0)
