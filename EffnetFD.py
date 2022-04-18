
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from tensorflow import keras
import sklearn.model_selection as model
import torch
import tensorflow as tf
from os.path import exists


def get_result_plot(fitted_model):
    history_pd = pd.DataFrame(fitted_model.history)

    plt.plot(history_pd['accuracy'])
    plt.plot(history_pd['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train set','test set'],loc='upper right')
    plt.show()


# Hyper-parameters
lr = 0.001
epochs = 15
dropout = 0.3
batch_size = 50

arr = np.load('./imageData.npz',allow_pickle=True)
img_list = arr['arr_0']
annotation_list = arr['arr_1']

x_train,x_tot,y_train,y_tot = model.train_test_split(img_list,annotation_list,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test = model.train_test_split(x_tot,y_tot,test_size=0.5,random_state=42)

device = torch.device("cuda" if torch.cuda.is_available()else "cpu")

if exists("./weights.h5"):
    print("hello")

backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
    weights = "imagenet",
    input_shape=(224,224,3),
    include_top = False
)
model = tf.keras.Sequential([
    backbone,
    keras.layers.Dense(1000),
    keras.layers.ReLU(),
    keras.layers.Dense(500),
    keras.layers.Dropout(dropout),
    keras.layers.GlobalAvgPool2D(),
    keras.layers.LayerNormalization(),
    keras.layers.Dense(1)]
)


model.summary()
loss =tf.keras.losses.BinaryFocalCrossentropy()
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./effnet_weights.h5",
                                                               save_weights_only=True,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
              loss = loss,
              metrics=['accuracy'])

temp = model.fit(x = x_train,
                 y = y_train,
                 validation_data = (x_val,y_val),
                 batch_size = batch_size,
                 epochs=epochs,
                 callbacks=[model_checkpoint_callback])
model.save("effnet_weights.h5")

temp2 = model.evaluate(x_test,
                       y_test,
                       batch_size=batch_size)

get_result_plot(model)