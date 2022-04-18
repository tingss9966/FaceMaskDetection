import csv

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import sklearn.model_selection as model
import torch
from torch import nn
import tensorflow as tf

dropout = 0.3
arr = np.load('./imageData.npz',allow_pickle=True)
img_list = arr['arr_0']
annotation_list = arr['arr_1']

x_train,x_tot,y_train,y_tot = model.train_test_split(img_list,annotation_list,test_size=0.3,random_state=42)
x_val,x_test,y_val,y_test = model.train_test_split(x_tot,y_tot,test_size=0.5,random_state=42)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
# m = keras_efficientnet_v2.EfficientNetV2M(pretrained="imagenet",input_shape=(None,None,3),classifier_activation='softmax',dropout=dropout)
# m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# print(len(x_train))
# m = applications.efficientnetv2()


# for parameters in model_vit.parameters():
#     parameters.requires_grad = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1000,200)
        self.fc2 = nn.Linear(200, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self,x):
        x = self.fc(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


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
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./weights",
                                                               save_weights_only=True,
                                                               monitor='val_accuracy',
                                                               mode='max',
                                                               save_best_only=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss = loss, metrics=['accuracy'])
temp = model.fit(x = x_train, y = y_train, validation_data = (x_val,y_val),batch_size = 50, epochs=2,callbacks=[model_checkpoint_callback])
temp2 = model.evaluate(x_test, y_test, batch_size=50)
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)
