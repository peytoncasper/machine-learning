import keras2onnx
import keras as keras
import numpy as np
import tensorflow as tf


def main():
    (training_images, training_labels) = load_data(
        "data/train-images-idx3-ubyte",
        "data/train-labels-idx1-ubyte",
        60000
    )

    (test_images, test_labels) = load_data(
        "data/t10k-images-idx3-ubyte",
        "data/t10k-labels-idx1-ubyte",
        10000
    )


    # Greyscale Images
    training_images = training_images / 255.0



    # 1. Flatten the 28x28 array into an input of 784 pixels
    # 2. Create a fully connected neuron layer of 128 neurons that uses the ReLU activation function
    # 3. Create a fully connected neuron layer of 10 neurons that maps the output of the previous layer
    # to a probability distribution of 10 outputs that add up to 1
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=5)

    loss = model.evaluate(test_images, test_labels)

    print("Model Loss: {}, Model Accuracy: {}".format(loss[0], loss[1]))

    convert_model(model)

def convert_model(model):
    minst_onnx = keras2onnx.convert_keras(model, "minst", doc_string="MINST Model")
    keras2onnx.save_model(minst_onnx, "minst.onnx")

def load_data(image_file_path, label_file_path, count):
    image_file = open(image_file_path, "rb")
    label_file = open(label_file_path, "rb")

    # Initialize array for 28x28 pixel image
    images = np.zeros([count, 28, 28])

    # Initialize array for 10 labels
    labels = np.zeros([count])

    # Iterate and remove header details
    image_file.read(16)
    label_file.read(8)

    # Iterate N elements from the dataset
    for i in range(count):
        # Convert from binary and mark the specified label for this sequence
        lbl = ord(label_file.read(1))
        labels[i] = lbl

        # Iterate and convert 784 pixels into 28x28 array
        for j in range(28):
            for k in range(28):
                pixel = image_file.read(1)
                images[i][j][k] = ord(pixel)

    image_file.close()
    label_file.close()

    return images, labels
