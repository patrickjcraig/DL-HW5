# Requirements and goals for this homework

# Train the given CNN with 700 digits (500 training and 200 for cross validation)

# Using the EMNIST, improve this baseline performance of the CNN by using the well-trained AlexNet, freezing 
# its weights, and striping the classification layers. 
# Explain how you selected the hyper parameters and show the performance in a confusion matrix.

# 1. Use the digits as inputs to AlexNet and train their targets.
# 2. Please implement the two basic methods for fine tuning
#   a. Depth augmented and width augmented architectures
# 3. Cross validate with 200 digits
# 4. Fine tune the extra layer and classifier with the 700 digit training set.
# 5. Use one or two layer classifier
# Six. Test the performance of the AlexNet with transfer learning in 300 test digits and prsent in a confusion matrix
# 7. Compare the AlexNet performance with the original CNN trained for the task in the small 1,000 digits dataset
#   a. Discuss the results and the reasons for the performance difference

# Import the dataset
from json import load
import numpy as np
from TensorFlow.networks_tf import AlexNet, Patricks_Smallish_CNN
from utils import load_data, split_data, load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
TF_ENABLE_ONEDNN_OPTS=0

def plot_learning_curves(history, model_name):
    plt.figure(figsize=(12, 4))
    plt.suptitle(f"Learning Curves for {model_name}")
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def train(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, LR_SCHED=False, learning_rate=1e-4):
    
    # Define tensorflow learning rate scheduler, optimizer, and early stopping
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 10))
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model with or without learning rate scheduler
    callbacks = [lr_scheduler, early_stopping] if LR_SCHED else [early_stopping]
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), callbacks=callbacks)
    
    # Print best validation accuracy
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy'])}")
    
    # evaluate model on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Predict and create confusion matrix
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    
    return model, history, test_accuracy, y_pred

if __name__ == '__main__':
    # Load the dataset
    inputs, labels = load_data('digits.npz') 
    
    # Scale and normalize the data
    inputs = inputs / 255.0
    inputs = np.expand_dims(inputs, axis=-1)
    
    x_train, y_train, x_val, y_val, X_test, y_test = split_data(inputs, labels) # Split the dataset
   
    # Define baseline model
    vanilla_model = Patricks_Smallish_CNN(input_shape=(28, 28, 1))
    vanilla_model.summary()
   
    # Define model head
    model_head = load_model(AlexNet(), r'TensorFlow/AlexNet_pretrained.h5') # Load the pretrained model
    
    # Remove unneeded layers
    for i in range(5):
        model_head.pop()
        
    # Freeze alexnet model weights
    for layer in model_head.layers:
        layer.trainable = False
    
    # Add dense layer augmentation units
    model_head.add(tf.keras.layers.Dense(2304, activation='relu'))
    model_head.add(tf.keras.layers.Dense(784, activation='relu'))
    model_head.add(tf.keras.layers.Reshape(target_shape=(28, 28, 1)))
    
    # Save model head for laters
    model_head.summary()
    
    # ----------------------------- Train the models -----------------------------
    epochs=50
    
    #  Train base model
    vann_model, vann_hist, vann_test_accuracy, vann_y_pred = train(vanilla_model, x_train, y_train, x_val, y_val, epochs=epochs)
    print("Base model trained") 
    
    # print accuracy
    print(f"Base Model Test Accuracy: {vann_model.evaluate(X_test, y_test)[1]}")
    
    # ---------------------------- Train depth augmented model -------------------
    model_base = Patricks_Smallish_CNN(input_shape=(28, 28, 1))
    
    depth_model = tf.keras.models.Sequential([model_head, model_base])
    depth_model.summary()
    
    # Compile model
    depth_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Check trainable parameters
    print(f"Trainable parameters: {len(model_head.trainable_variables)}")
    
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train depth model
    depth_initial_fit = depth_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping])
    print("Depth model trained")
    
    # Calculate test accuracy for depth model
    print(f"Depth Model Test Accuracy: {depth_model.evaluate(X_test, y_test)[1]}")
    
    # Predict and create confusion matrix for depth model
    depth_y_pred = depth_model.predict(X_test)
    depth_y_pred = np.argmax(depth_y_pred, axis=1)

    # ---------------------------- Train width augmented model -------------------
    # Feature extractor
    feature_extractor = load_model(AlexNet(), r'TensorFlow/AlexNet_pretrained.h5')

    # pop the last 5 layers of alexnet
    for i in range(5):
        feature_extractor.pop()
    
    # define classifier
    classifier = Patricks_Smallish_CNN(input_shape=(28,28,1))
    
    # Branch 1
    main_branch = tf.keras.models.Sequential([
        feature_extractor,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(392,activation='relu'),
    ], name='main_branch')
    
    # Branch 2
    width_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(2304, activation='relu'),
        tf.keras.layers.Dense(392, activation='relu'), 
    ], name = 'width_augmentation')
    
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=(28, 28, 1))
    
    main_branch_output = main_branch(input_layer)
    width_augmentation_output = width_augmentation(input_layer)
    
    # Concate outputs
    concatenated = tf.keras.layers.concatenate([main_branch_output, width_augmentation_output])
    
    # Reshape to fit CNN
    reshaped = tf.keras.layers.Reshape(target_shape=(28, 28, 1))(concatenated)
    
    # Connect the concatenated output to the classifier
    output = classifier(reshaped)
    
    # Define the model
    width_augmented_model = tf.keras.models.Model(inputs=input_layer, outputs=output)
    width_augmented_model.summary()
    
    # Compile the model
    width_augmented_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Print the model summary
    width_augmented_model.summary()

    # Compile the model
    width_augmented_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    width_initial_fit = width_augmented_model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val), callbacks=[early_stopping])
    
    # Calculate test accuracy for width model
    print(f"Width Model Test Accuracy: {width_augmented_model.evaluate(X_test, y_test)[1]}")
    
    # Predict and create confusion matrix for width model
    width_y_pred = width_augmented_model.predict(X_test)
    width_y_pred = np.argmax(width_y_pred, axis=1)
    
    # ---------------------------- Plotting results ------------------------------
    plot_learning_curves(vann_hist, "Base Model")
    plot_confusion_matrix(y_test, vann_y_pred, "Base Model")
    
    plot_learning_curves(depth_initial_fit, "Depth Augmented Model")
    plot_confusion_matrix(y_test, depth_y_pred, "Depth Augmented Model")
    
    plot_learning_curves(width_initial_fit, "Width Augmented Model")
    plot_confusion_matrix(y_test, width_y_pred, "Width Augmented Model")
    
    # Printing of test accuracies
    print(f"Base Model Test Accuracy: {vann_test_accuracy}")
    print(f"Depth Model Test Accuracy: {depth_model.evaluate(X_test, y_test)[1]}")
    print(f"Width Model Test Accuracy: {width_augmented_model.evaluate(X_test, y_test)[1]}")
    
