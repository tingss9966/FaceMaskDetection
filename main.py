import csv

import keras.layers
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model
import sklearn as sk
from torchvision import models
import torch
from keras import applications
from torch import nn
import keras_efficientnet_v2
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

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

model_vit = keras_efficientnet_v2.EfficientNetV2M(pretrained="imagenet")
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


backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2M(
    weights = "imagenet",
    input_shape=(224,224,3),
    include_top = False
)
model = tf.keras.Sequential([
    backbone,
    keras.layers.GlobalAvgPool2D(),
    keras.layers.Dense(1)]
)


model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss = tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])
temp = model.fit(x = x_train, y = y_train, validation_data = (x_val,y_val),batch_size = 30, epochs=15)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
#
# model.to(device)
#
# x_val = x_val.reshape((-1,3,224,224))
# x_val = torch.Tensor(x_val)
# y_val = torch.Tensor(y_val)
# val_data = TensorDataset(x_val,y_val)
# val_dataloader = DataLoader(val_data)
#
#
# def validation(model, validateloader, criterion):
#     val_loss = 0
#     accuracy = 0
#
#     for images, labels in iter(validateloader):
#         images, labels = images.to('cpu'), labels.to('cpu',dtype=torch.int64)
#         images.view(images.shape[0],-1)
#         output = model.forward(images)
#         val_loss += criterion(output, labels).item()
#
#         probabilities = torch.exp(output)
#
#         equality = (labels.data == probabilities.max(dim=1)[1])
#         accuracy += equality.type(torch.FloatTensor).mean()
#
#     return val_loss, accuracy
#
#
#
# epochs = 15
# steps = 0
# print_every = 40
#
# x_train = x_train.reshape((2850,3,224,224))
# tensorx = torch.Tensor(x_train)
# tensory = torch.Tensor(y_train)
# data = TensorDataset(tensorx,tensory)
# dataloader = DataLoader(data)
#
# print(x_train.shape)
# for e in range(epochs):
#     model.train()
#     running_loss = 0
#     for images, labels in iter(dataloader):
#         images = images.to('cpu', dtype=torch.long)
#         labels = labels.to('cpu', dtype=torch.long)
#         steps += 1
#         optimizer.zero_grad()
#         print(images.shape)
#         output = model.forward(images)
#         print(output)
#         loss = criterion(output.squeeze(), labels.squeeze())
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         model.eval()
#
#         # Turn off gradients for validation, saves memory and computations
#         with torch.no_grad():
#             validation_loss, accuracy = validation(model, val_dataloader, criterion)
#
#         print("Epoch: {}/{}.. ".format(e + 1, epochs),
#               "Training Loss: {:.3f}.. ".format(running_loss / print_every),
#               "Validation Loss: {:.3f}.. ".format(validation_loss / len(val_dataloader)),
#               "Validation Accuracy: {:.3f}".format(accuracy / len(val_dataloader))
#               )
#
#         running_loss = 0
#         model.train()
