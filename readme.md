# Rock-Paper-Scissors Image Classification Model

## Overview

This project, focuses on classifying images into rock, paper, or scissors categories using a neural network. The aim is to demonstrate the effectiveness of a simple neural network model on image classification tasks.

## Data Preparation

- The dataset consists of hand gesture images sized at 300x200x3 pixels, classified into rock, paper, and scissors categories.
- We converted the images to grayscale to reduce computational complexity without significantly impacting accuracy, Changing the images from 300x200x3 to 300x200.
- The dataset is split into 80% for training and 20% for testing.

## Model Architecture

- The neural network model consists of an input layer with 60,000 nodes (flattened grayscale images), a hidden layer with 5 nodes (ReLU activation), and an output layer with 3 nodes (SoftMax function).
- Training was conducted over 100 epochs with a learning rate of 0.001, using cross-entropy as the loss function.
- The model achieved an accuracy of 76%, while a simpler SoftMax model achieved 73% accuracy.

## Instructions

1. **Download Dataset**:

   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors).
   - Extract and rename the folder from `archive` to `dataset`.

2. **Model Training**:

   - Run `python rps_nn_model.py` or `python rps_softmax_model.py` to train the model.

3. **Testing the Model**:
   - Execute `python test_model.py` to test the model on sample images from the `pics` subfolder.

## TODO:

We plan to implement a more advanced model, such as a convolutional neural network, in our final project for improved accuracy in image classification tasks.

## Contributors

- Itamar Cohen
- Eliyahu Greenblatt
- Arad Ben Menashe

---

_This project is part of an academic assignment and is meant for educational purposes only._
