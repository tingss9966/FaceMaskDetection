import numpy as np
import cv2 as cv
from FaceMask import *
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from ISR.models import RDN
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

def load_annotation():
    tree = ET.parse(annotation_path+"maksssksksss0.xml")
    root = tree.getroot()
    filename = root.find('filename').text
    image = np.array(load_image(filename))
    lst = []
    if image is not None:
        for index in root.iter('object'):
            fcmsk = FaceMask()
            fcmsk.set_mask(index.find('name').text == "with_mask")
            xmin = int(index.find('bndbox').find('xmin').text)
            xmax = int(index.find('bndbox').find('xmax').text)
            ymin = int(index.find('bndbox').find('ymin').text)
            ymax = int(index.find('bndbox').find('ymax').text)
            cropped_image = image[ymin:ymax,xmin:xmax]
            fcmsk.set_image(cropped_image)
            lst.append(fcmsk)
    return lst

lst = load_annotation()

for i in lst:
    plt.imshow(i.image)
    plt.show()
    print(i.mask)

# sr = cv.dnn_superres.DnnSuperResImpl_create()
# path = "LapSRN_x8.pb"
# sr.readModel(path)
# sr.setModel("lapsrn",8)
# result = sr.upsample(lst[0].image)
# plt.imshow(result)
# plt.show()
#

# rdn = RDN(weights='psnr-small')
# sr_img = rdn.predict(lst[0].image)
# plt.imshow(sr_img)
# plt.show()