import cv2 as cv
import numpy as np
from tensorflow import keras

labels_dict={0:'without_mask',1:'with_mask'}
face_clsfr=cv.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4
load_model = keras.models.load_model('./effnet_weights.h5')
cv.namedWindow("Face Mask Detection")
cam = cv.VideoCapture(0)
color_dict={0:(0,0,255),1:(0,255,0)}

sr = cv.dnn_superres.DnnSuperResImpl_create()
path = "models/LapSRN_x8.pb"
sr.readModel(path)
sr.setModel("lapsrn", 8)

classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    rval, image = cam.read()
    image=cv.flip(image,1,1)
    small = cv.resize(image, (image.shape[1] // size, image.shape[0] // size))
    faces = classifier.detectMultiScale(small)
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = image[y:y + h, x:x + w]
        # face_img = sr.upsample(face_img)
        resized = cv.resize(face_img, (224, 224))
        reshaped = np.reshape(resized, (1, 224, 224, 3))
        reshaped = np.vstack([reshaped])
        result = load_model.predict(reshaped)
        result = np.array(result)
        result = result.reshape((1))
        result = result.clip(0,1)
        label = ""
        if result[0] < 0.5:
            label = 0
        else:
            label = 1
        cv.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
        cv.rectangle(image, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv.putText(image, labels_dict[label], (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    if image is not None:
        cv.imshow('Face Mask Detection', image)
    key = cv.waitKey(10)
    if key == 27:
        break
cam.release()
cv.destroyAllWindows()