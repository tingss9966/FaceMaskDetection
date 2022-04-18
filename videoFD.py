import cv2
import numpy as np
from tensorflow import keras

labels_dict={0:'without_mask',1:'with_mask'}
face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
size = 4
load_model = keras.models.load_model('./effnet_weights.h5')
cv2.namedWindow("Face Mask Detection")
cam = cv2.VideoCapture(0)
color_dict={0:(0,0,255),1:(0,255,0)}

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    rval, image = cam.read()
    image=cv2.flip(image,1,1)
    small = cv2.resize(image, (image.shape[1] // size, image.shape[0] // size))
    faces = classifier.detectMultiScale(small)
    for f in faces:
        (x, y, w, h) = [v * size for v in f]
        face_img = image[y:y + h, x:x + w]
        resized = cv2.resize(face_img, (224, 224))
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
        cv2.rectangle(image, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(image, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(image, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

    if image is not None:
        cv2.imshow('Face Mask Detection', image)
    key = cv2.waitKey(10)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()