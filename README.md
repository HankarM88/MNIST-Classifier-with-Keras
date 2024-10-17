# MNIST Classifier with Keras

This project demonstrates how to build a deep learning model using **Keras** to classify handwritten digits from the **MNIST dataset**. The MNIST dataset consists of grayscale images of handwritten digits (0-9) and is widely used for training various image processing systems.

## Dataset
The model was trained on the **MNIST dataset**, which can be downloaded from Kaggle:  
[MNIST Dataset on Kaggle](https://www.kaggle.com/c/digit-recognizer/data)

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction
The goal of this project is to classify images of handwritten digits using a **Convolutional Neural Network (CNN)**. The CNN is trained to predict the correct digit for each image.

The process to build the model is as follows:
1. **Load the dataset**
2. **Prepare data for training:**
   - Reshape the training and test data
   - Normalize the features
3. **Split the data** into training and validation sets
4. **Build the CNN model**
5. **Train the model** and make predictions
6. **Submit the predictions** into a dataframe
7. **Save the model**

## Model Architecture
The CNN architecture consists of:
- Convolutional layers to extract features from the images
- Max-pooling layers to downsample the feature maps
- Fully connected layers to classify the digits

### Steps to Build the Model:
1. Input layer: 28x28 grayscale images (reshaped)
2. 2D Convolutional layers with ReLU activation
3. MaxPooling layers
4. Dropout layers for regularization
5. Fully connected (Dense) layers
6. Output layer: Softmax for multi-class classification (0-9 digits)

## Installation

To run this project, you will need the following libraries:
- Python 3.x
- Keras
- TensorFlow
- NumPy
- Pandas
- Matplotlib (for visualizations)

You can install the required dependencies using the following command:

```bash
pip install keras tensorflow numpy pandas matplotlib
```
## Usage
1. Load and Prepare Data
 ```python
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```
2. Build CNN Model
 ```python
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
```
3. Make predictions and evaluate the model
```python
  predictions = model.predict(X_test)
  accuracy = model.evaluate(x_test, y_test)
  ```
4. Save the model
```python
 model.save("mnistModel2.h5")
  ```
## Results 
The model achieved a high accuracy on the test dataset. Below is a table summarizing the model's performance:
**Accuracy:**  99%
**Loss:** 0.05
## Conclusion
This project demonstrates the power of Convolutional Neural Networks (CNNs) in image classification tasks. The MNIST dataset serves as a good starting point for learning how to implement deep learning models in Keras.
