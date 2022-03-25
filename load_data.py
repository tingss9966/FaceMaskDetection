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


def load_image(filename):
    image = cv.imread(image_path+filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if image is not None:
        return image
    return None


def load_annotation(super_res = False):
    lst = []
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "models/LapSRN_x8.pb"
    sr.readModel(path)
    sr.setModel("lapsrn", 8)
    for filename in os.listdir(annotation_path):
        f = os.path.join(annotation_path, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        filename = root.find('filename').text
        image = np.array(load_image(filename))
        if image is not None:
            for index in root.iter('object'):
                fcmsk = FaceMask()
                fcmsk.set_mask(index.find('name').text == "with_mask")
                xmin = int(index.find('bndbox').find('xmin').text)
                xmax = int(index.find('bndbox').find('xmax').text)
                ymin = int(index.find('bndbox').find('ymin').text)
                ymax = int(index.find('bndbox').find('ymax').text)
                cropped_image = image[ymin:ymax,xmin:xmax]
                if super_res:
                    result = sr.upsample(cropped_image)
                    fcmsk.set_image(result)
                else:
                    fcmsk.set_image(cropped_image)
                    pass
                lst.append(fcmsk)
    return lst


lst = load_annotation(super_res=True)
print("finished")
