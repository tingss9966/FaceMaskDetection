import csv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as model
import sklearn as sk
from torchvision import models
import torch
import keras_efficientnet_v2

# arr = np.load('./imageData.npz',allow_pickle=True)
# img_list = arr['arr_0']
# annotation_list = arr['arr_1']
#
# x_train,x_tot,y_train,y_tot = model.train_test_split(img_list,annotation_list,test_size=0.3,random_state=42)
# x_val,x_test,y_val,y_test = model.train_test_split(x_tot,y_tot,test_size=0.5,random_state=42)
#
# device = torch.device("cuda" if torch.cuda.is_available()
#                                   else "cpu")
# print(torch.cuda.is_available())
# models.efficientnet_v2_m()

