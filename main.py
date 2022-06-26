import scipy.io as sio
import numpy as np
import keras
import tensorflow
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from nl_cnn import create_nl_cnn_model

if __name__ == "__main__":
    train_set = sio.loadmat("emo_dataset.mat")

    print(train_set)

    x_train = train_set["Samples"]
    y_train = train_set["Labels"]
    y_train = np.reshape(y_train, y_train.size)



    x_train = np.reshape(x_train, [np.shape(x_train)[0], np.shape(x_train)[1], np.shape(x_train)[2], 1])

    num_classes = np.max(y_train) + 1

    x_train = x_train.astype("float32")
    y_train = y_train.astype('uint8')

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    input_shape = np.shape(x_train)[1:4]
    input_channels = np.shape(x_train)[3]

    print("x_train shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("input_shape: ", input_shape)
    print("input_channels: ", input_channels)

    model = create_nl_cnn_model(input_shape, num_classes, k=2, separ=0, flat=0, width=32, nl=(1,0), add_layer=0)
    model.summary()
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
    tensorflow.keras.utils.plot_model(model, "model.png" , show_shapes=True, show_layer_names=True)
    history = model.fit(x_train, y_train,
                        batch_size=5,
                        epochs=120,
                        verbose=1,  # aici 0 (nu afiseaza nimic) 1 (detaliat) 2(numai epocile)
                        validation_split=0.2)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    model.save("emo-db-model-windows")
    score = model.evaluate(x_test, y_test, verbose=1)
    print(score)

