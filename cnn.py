import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
import sklearn.model_selection as model
import numpy as np
import keras
import torch
import tensorflow as tf


def get_result_plot(fitted_model):
    history_pd = pd.DataFrame(fitted_model.history)
    plt.plot(history_pd['accuracy'])
    plt.plot(history_pd['val_accuracy'])
    plt.title('Model accuracy CNN')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train set', 'validation set'], loc='upper right')
    plt.show()


# Hyper-parameters
lr = 0.00001
epochs = 30
dropout = 0.3
batch_size = 50

arr = np.load('./imageData.npz', allow_pickle=True)
img_list = arr['arr_0']
annotation_list = arr['arr_1']
ann = []

for a in annotation_list:
    if a:
        ann.append(1)
    else:
        ann.append(0)
ann = np.array(ann)
x_train, x_tot, y_train, y_tot = model.train_test_split(img_list, ann, test_size=0.3,
                                                        random_state=42)
x_val, x_test, y_val, y_test = model.train_test_split(x_tot, y_tot,
                                                      test_size=0.5,
                                                      random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = keras.models.Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(224, 224, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dropout(0.5),
    Dense(100, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./cnn_weights.h5",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

opt = tf.keras.optimizers.Adam(learning_rate=lr, decay=1e-5)
model.compile(optimizer=opt, loss=tf.keras.losses.BinaryFocalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epochs,
                    validation_data=(x_val, y_val), batch_size=batch_size,
                    callbacks=[model_checkpoint_callback, early_stop])

best_model = keras.models.load_model("cnn_weights.h5")

temp2 = best_model.evaluate(x_test,
                            y_test,
                            batch_size=batch_size)
get_result_plot(history)

#
