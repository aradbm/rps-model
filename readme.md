# Rock-Paper-Scissors Image Classification Model

## Overview

This project focuses on classifying images of hand gestures into rock, paper, or scissors categories using neural network models. We demonstrate the effectiveness of both simple and advanced neural network models, including a CUDA-compatible Convolutional Neural Network (CNN) with dropout and data augmentation, on image classification tasks.

[Project Instructions](metadata/Project_Instructions.pdf) & [Detailed Project Book](RPS_Project_Book.pdf)

![cnn_model](metadata/cnn_model.png)

## Kaggle Notebook

- You can find the CUDA-compatible CNN model and its running on Kaggle here: [[Rock-Paper-Scissors Classification - 97.9% Acc](https://www.kaggle.com/code/aradbenmenashe/rock-paper-scissors-classification-97-9-acc)]

## Performance

The following table summarizes the performance of the CNN model under various configurations:

| No. | Epochs | Augmented Data | Learning Rate | Drop Out | Final Accuracy % |
| --- | ------ | -------------- | ------------- | -------- | ---------------- |
| 1   | 100    | T              | 0.0001        | X        | 93.6             |
| 2   | 100    | T              | 0.001         | 0.5      | 95.6             |
| 3   | 100    | F              | 0.001         | 0.5      | 92.9             |
| 4   | 50     | T              | 0.001         | 0.3      | 97.94            |
| 5   | 50     | T              | 0.001         | X        | 92.92            |
| 6   | 50     | F              | 0.001         | 0.3      | 93.83            |
| 7   | 50     | F              | 0.001         | X        | 92.46            |

<p align="center">
   <img src="metadata/output1.png" width=300/>
</p>

## Data Preparation

- The dataset consists of hand gesture images sized at 300x200x3 pixels, classified into rock, paper, and scissors categories.
- We converted the images to grayscale to reduce computational complexity without significantly impacting accuracy. This changes the images from 300x200x3 to 300x200.
- The dataset is split into 80% for training and 20% for testing.

## Models Architecture

### Simple Neural Network Model

- Input layer with 60,000 nodes (flattened grayscale images), a hidden layer with 5 nodes (ReLU activation), and an output layer with 3 nodes (SoftMax function).
- Trained over 100 epochs with a learning rate of 0.001, using cross-entropy as the loss function.
- Achieved an accuracy of 76%, while a simpler SoftMax model achieved 73% accuracy.

### Convolutional Neural Network Model

- The CNN model, defined in `CNNModel`, employs an architecture for efficient image classification.
- It consists of two convolutional layers: the first with 8 filters and the second with 16 filters, both using a kernel size of 3. Each convolutional layer is followed by max pooling for feature reduction.
- The network uses ReLU activation functions and incorporates a two-layer fully connected network, reducing the dimension from 16 _ 75 _ 50 to 256, and finally to 3 for classification.
- Dropout can be configured during training, e.g., `dp_rate = 0.3; model = CNNModel(dropout_rate=dp_rate).to(device)`.
- Data augmentation is applied during training, including random rotation and flipping of images.
- The model is CUDA-compatible, enabling training on GPU environments for faster computation.
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
   - For training the CNN model with CUDA, run `python train_cnn_model_w_cuda.py`.

3. **Testing the Model**:
   - To test the simple neural network model, execute `python test_nn_model.py`.
   - To test the CNN model, execute `python test_cnn_model.py`.

## Contributors

- Itamar Cohen
- Eliyahu Greenblatt
- Arad Ben Menashe

---

_This project is part of an academic assignment and is meant for educational purposes only._
