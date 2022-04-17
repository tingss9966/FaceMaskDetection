import numpy as np
import cv2 as cv
from FaceMask import *
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
from PIL import Image
# import torch

image_path = "./dataset/images/"
annotation_path = "./dataset/annotations/"
tester = "./Tester/"


def load_image(filename):
    image = cv.imread(image_path+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if image is not None:
        return image
    return None


def load_annotation(dir_path, super_res = False):
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "models/LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)
    annotaion = []
    images = []
    counter =0
    for filename in os.listdir(dir_path):
        f = os.path.join(dir_path, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        filename = root.find('filename').text
        image = np.array(load_image(filename))
        if image is not None:
            for index in root.iter('object'):
                annotaion.append(index.find('name').text == "with_mask")
                xmin = int(index.find('bndbox').find('xmin').text)
                xmax = int(index.find('bndbox').find('xmax').text)
                ymin = int(index.find('bndbox').find('ymin').text)
                ymax = int(index.find('bndbox').find('ymax').text)
                cropped_image = image[ymin:ymax,xmin:xmax]
                if super_res:
                    result = sr.upsample(cropped_image)
                else:
                    result = cropped_image.reshape(cropped_image.shape[0],-1)
                # result = Image.fromarray(result)
                # result.save("./ProcessedImages/fsmsk" + str(counter) + ".png")
                images.append(result)
                counter+=1
    annotaion = np.array(annotaion)
    return images, annotaion


images, annotation = load_annotation(annotation_path,True)
images = np.asarray(images,dtype='object')
annotation = np.asarray(annotation)
np.savez('./imageData.npz',images,annotation, annotation)


