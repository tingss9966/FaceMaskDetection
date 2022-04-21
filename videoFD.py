"""Import packages and libraries"""
import cv2 as cv
import numpy as np
from tensorflow import keras

# Parameters: (Choose which classification model to use, Need to run model first before this)
mod = "EfficientNetV2"  # EfficientNetV2/InceptionNetV2/CNN

if mod == "EfficientNetV2":
    file = "./effnet_weights.h5"
elif mod == "InceptionNetV2":
    file = "./inception_weights.h5"
else:
    file = "./cnn_weights.h5"

# Cascade Classifier for face detection
classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'without_mask', 1: 'with_mask'}
size = 4
# load model for classification
load_model = keras.models.load_model(file)
cv.namedWindow("Face Mask Detection")
cam = cv.VideoCapture(0)

# Two different colors to differ wearing a mask or not
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}

# sr = cv.dnn_superres.DnnSuperResImpl_create()
# path = "models/LapSRN_x8.pb"
# sr.readModel(path)
# sr.setModel("lapsrn", 8)


while True:
    rval, image = cam.read()
    image = cv.flip(image, 1, 1)
    small = cv.resize(image, (image.shape[1] // size, image.shape[0] // size))
    faces = classifier.detectMultiScale(small)
    for f in faces:
        # Find the bounding box of the face
        (x, y, w, h) = [v * size for v in f]
        face_img = image[y:y + h, x:x + w]
        # face_img = sr.upsample(face_img)
        # Preprocess the image for prediction
        resized = cv.resize(face_img, (224, 224))
        reshaped = np.reshape(resized, (1, 224, 224, 3))
        reshaped = np.vstack([reshaped])
        # predict the image given
        result = load_model.predict(reshaped)
        result = np.array(result)
        result = result.reshape((1))
        result = result.clip(0, 1)
        label = ""
        # based on the results given, determine if wearing a mask or not
        if result[0] < 0.5:
            label = 'without_mask'
            color = (0, 0, 255)
        else:
            label = 'with_mask'
            color = (0, 255, 0)
        # Draw rectangle around the face and color it with the classified color
        cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv.rectangle(image, (x, y - 40), (x + w, y), color, -1)
        cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (255, 255, 255), 2)

    if image is not None:
        cv.imshow('Face Mask Detection', image)
    # to exit the camera feed
    key = cv.waitKey(10)
    if key == 27:
        break
cam.release()
cv.destroyAllWindows()
