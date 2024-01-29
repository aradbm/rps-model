# Rock-Paper-Scissors Image Classification Model

## Overview

This project focuses on classifying images of hand gestures into rock, paper, or scissors categories using neural network models. We demonstrate the effectiveness of both simple and advanced neural network models on image classification tasks.

## Data Preparation

- The dataset consists of hand gesture images sized at 300x200x3 pixels, classified into rock, paper, and scissors categories.
- We converted the images to grayscale to reduce computational complexity without significantly impacting accuracy. This changes the images from 300x200x3 to 300x200.
- The dataset is split into 80% for training and 20% for testing.

## Model Architecture

### Simple Neural Network Model
- Input layer with 60,000 nodes (flattened grayscale images), a hidden layer with 5 nodes (ReLU activation), and an output layer with 3 nodes (SoftMax function).
- Trained over 100 epochs with a learning rate of 0.001, using cross-entropy as the loss function.
- Achieved an accuracy of 76%, while a simpler SoftMax model achieved 73% accuracy.

### Convolutional Neural Network Model
- The CNN model, defined in `LightCNNModel`, employs a lightweight architecture for efficient image classification.
- It consists of two convolutional layers: the first with 8 filters and the second with 16 filters, both using a kernel size of 3. Each convolutional layer is followed by max pooling for feature reduction.
- The network uses ReLU activation functions and incorporates a two-layer fully connected network, reducing the dimension from 16 * 75 * 50 to 256, and finally to 3 for classification.
- The model is initialized with Xavier uniform initialization for both convolutional and fully connected layers, ensuring optimal weights at the start.
- Seed-based initialization is used to ensure reproducibility of the model's performance.
- This architecture demonstrates improved accuracy over simpler neural network models, making it more suitable for complex image classification tasks like distinguishing between rock, paper, and scissors hand gestures.


## Instructions

1. **Download Dataset**:

   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).
   - Extract and rename the folder from `archive` to `dataset`.

2. **Model Training**:

   - For training the simple NN model, run `python train_nn_model.py`.
   - For training the SoftMax model, run `python train_softmax_model.py`.
   - For training the CNN model, run `python train_cnn_model.py`.

3. **Testing the Model**:
   - To test the simple neural network model, execute `python test_nn_model.py`.
   - To test the CNN model, execute `python test_cnn_model.py`.

## Contributors

- Itamar Cohen
- Eliyahu Greenblatt
- Arad Ben Menashe

---

_This project is part of an academic assignment and is meant for educational purposes only._
