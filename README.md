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

