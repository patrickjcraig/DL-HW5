from tensorflow.keras import layers, models
import numpy as np

# An example CNN model
def ExampleCNN():
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=(4, 4), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(4, 4), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))
    return model

def Patricks_Smallish_CNN(input_shape=(3, 3, 256)):
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size=(2, 2), activation='relu', input_shape=input_shape, padding='same'))  
    model.add(layers.Conv2D(16, kernel_size=(1, 1), activation='relu', padding='same'))  
    model.add(layers.Flatten())  

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10,activation='softmax'))  
    return model


# Pre-trained AlexNet structure
def AlexNet():
    model = models.Sequential()

    # Feature extraction layers
    model.add(layers.Conv2D(64, (5, 5), strides=1, activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(192, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Conv2D(384, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))

    model.add(layers.Flatten())

    # Classifier layers
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dense(26))

    return model


if __name__ == '__main__':
    '-----Load the digit dataset-----'
    digit_data = np.load('digits.npz')
    inputs = digit_data['inputs']
    labels = digit_data['labels']

    '-----Load the pre-trained AlexNet model-----'
    model = AlexNet()
    model.load_weights('AlexNet_pretrained.h5')

