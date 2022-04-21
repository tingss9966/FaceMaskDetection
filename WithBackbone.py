"""import all libraries and packages"""
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from tensorflow import keras
import sklearn.model_selection as model
import torch
import tensorflow as tf
import time

# Hyper-parameters, Global variables
lr = 0.001
epochs = 30
dropout = 0.3
batch_size = 50


#Time history class, used to record the time of each epoch
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


"""
Plot out the results of a fitted model
fitted_model: the return statement of a model.fit
"""
def get_result_plot(fitted_model):
    history_pd = pd.DataFrame(fitted_model.history)
    plt.plot(history_pd['accuracy'])
    plt.plot(history_pd['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train set', 'validation set'], loc='upper right')
    plt.show()


"""Train a model with given training set and validation set, and specified backbone
x_train: Training set data
y_train: Training set label
 x_val: Validation set data
 y_val: Validation set label
 x_test: Test set Data
 y_test: Test set label
 backbonename: specified backbone
"""
def NetTrain(x_train, y_train, x_val, y_val,x_test,y_test, backbonename):
    # Choose between two model as the backbone of the fine tuned model
    if backbonename == "EfficientNetV2":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
            weights="imagenet",
            input_shape=(224, 224, 3),
            include_top=False
        )
        model_file_name = "./effnet_weights.h5"
    else:
        backbone = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3)
        )
        model_file_name = "./inception_weights.h5"
    # create a fined tuned model with specified balcbone
    model = tf.keras.Sequential([
        backbone,
        keras.layers.Dense(1000),
        keras.layers.ReLU(),
        keras.layers.Dense(500),
        keras.layers.Dropout(dropout),
        keras.layers.GlobalAvgPool2D(),
        keras.layers.LayerNormalization(),
        keras.layers.Dense(1)]
    )

    model.summary()
    loss = tf.keras.losses.BinaryFocalCrossentropy()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    # Only save the best model with the highest validation accuracy to file
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_name,
                                                                   monitor='val_accuracy',
                                                                   mode='max',
                                                                   save_best_only=True)
    # compile the model with Adam optimizer
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=loss,
                  metrics=['accuracy'])

    time_callback = TimeHistory()
    # Train the model
    history = model.fit(x=x_train,
                        y=y_train,
                        validation_data=(x_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[model_checkpoint_callback, time_callback, early_stop])
    times = time_callback.times

    # load saved model for evaluation on test set
    best_model = keras.models.load_model(model_file_name)

    # evaluate test set
    test = best_model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size)

    return history, times, test


if __name__ == "__main__":
    # Load data after processing
    arr = np.load('./imageData.npz', allow_pickle=True)
    img_list = arr['arr_0']
    annotation_list = arr['arr_1']

    # Split the data into training validation and test set.
    x_train, x_tot, y_train, y_tot = model.train_test_split(img_list, annotation_list, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = model.train_test_split(x_tot, y_tot, test_size=0.5, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run both models and save the best model of each to file
    eff_history, eff_time, eff_test = NetTrain(x_train, y_train, x_val, y_val, x_test, y_test, "EfficientNetV2")
    get_result_plot(eff_history)
    inc_history, inc_time, inc_test = NetTrain(x_train, y_train, x_val, y_val, x_test, y_test, "InceptionResNetV2")
    get_result_plot(inc_history)

    len_inc = len(inc_time)
    len_eff = len(eff_time)
    x_inc = np.arange(len_inc)+1
    x_eff = np.arange(len_eff)+1

    # printing the running time of each epoch as well as the mean and total run time
    print(eff_time)
    print(inc_time)
    print("Mean training time EfficientNetV2" + str(np.mean(np.array(eff_time))))
    print("Mean training time InceptionResNetV2" + str(np.mean(np.array(inc_time))))
    print("Total training time EfficientNetV2" + str(np.sum(np.array(eff_time))))
    print("Total training time InceptionResNetV2" + str(np.sum(np.array(inc_time))))

    # plot a graph comparing the running time of each model
    plt.plot(x_eff, eff_time)
    plt.plot(x_inc, inc_time)
    plt.xlabel("epochs")
    plt.ylabel("Time/Sec")
    plt.title("Training Time")
    plt.legend(["EfficientNetV2", "InceptionResNetV2"], loc='upper right')
    plt.show()

