import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def load_data(path):
    # Load the digit dataset from digits.npz
    digit_data = np.load(path)
    inputs = digit_data['inputs']
    labels = digit_data['labels']
    print(f"Dataset Loaded: {inputs.shape} {labels.shape}")
    return inputs, labels

# Split the dataset into training and cross validation sets
def split_data(inputs, labels):
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
    print(f"Training Set: {x_train.shape} {y_train.shape}")
    print(f"Validation Set: {x_val.shape} {y_val.shape}")
    print(f"Test Set: {X_test.shape} {y_test.shape}")
    print(f"Splitting Dataset Complete")
    return x_train, y_train, x_val, y_val, X_test, y_test


def load_model(model, weights, verbose=0):
    
    model = model # define model arch
    
    # Check for weights
    if weights:
        model.load_weights(weights)
        print(f"Model loaded with weights: {weights}")
        # print architecture

    else:
        print("Model loaded without weights")
        
    if verbose == 1:
        model.summary()
    else:
        None

    return model