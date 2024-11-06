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
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

# Load the digit dataset from digits.npz
digit_data = np.load('digits.npz')
inputs = digit_data['inputs']
labels = digit_data['labels']
