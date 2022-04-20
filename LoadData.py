"""Import all libraries and packages"""
import numpy as np
import cv2 as cv
import xml.etree.ElementTree as ET
import os
from PIL import Image

# Parameters (Tune if don't want to use super res)
super_res = True

# Image and Annotation paths
image_path = "./dataset/images/"
annotation_path = "./dataset/annotations/"


# A tester path used for testing
# tester = "./Temp/"

""" Load the image given the file name. Return the RGB image

    filename: str: the file name of the image
    """
def load_image(filename):
    image = cv.imread(image_path + filename)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if image is not None:
        return image
    return None


""" Read annotations and load the corresponding images to file
    dir_path: str: the directory path for the annotations
    super_res: bool: whether or not to use super resolution on image preprocessing
    """
def load_annotation(dir_path, super_res=False):
    # Image Super Resolution used to increase the resolution of images to get better training data
    sr = cv.dnn_superres.DnnSuperResImpl_create()
    path = "models/LapSRN_x8.pb"
    sr.readModel(path)
    # Here we us LapSRNx8
    sr.setModel("lapsrn", 8)
    annotation = []
    resized_images = []
    counter = 0
    # Iter through all the files and preprocess the images
    for filename in os.listdir(dir_path):
        # Parse through the data structure and find the bounding box of the face
        f = os.path.join(dir_path, filename)
        tree = ET.parse(f)
        root = tree.getroot()
        filename = root.find('filename').text
        image = np.array(load_image(filename))
        if image is not None:
            for index in root.iter('object'):
                xmin = int(index.find('bndbox').find('xmin').text)
                xmax = int(index.find('bndbox').find('xmax').text)
                ymin = int(index.find('bndbox').find('ymin').text)
                ymax = int(index.find('bndbox').find('ymax').text)
                cropped_image = image[ymin:ymax, xmin:xmax]
                # The option to use super resolution or not
                if super_res:
                    result = sr.upsample(cropped_image)
                else:
                    result = cropped_image.reshape(cropped_image.shape[0], -1)
                # Creating array of image and resize them to the same size for training
                result = Image.fromarray(result)
                result = result.resize((224, 224))
                result = np.array(result)
                annotation.append(index.find('name').text == "with_mask")
                resized_images.append(result)
                counter += 1
    annotation = np.array(annotation)
    return resized_images, annotation


if __name__ == "__main__":
    # Run the image loading algorithm and save it to a .npz file
    images, annotations = load_annotation(annotation_path, super_res)
    images = np.asarray(images)
    annotations = np.asarray(annotations)
    np.savez('./imageData.npz', images, annotations)
    print(len(images))
    print(len(annotations))
