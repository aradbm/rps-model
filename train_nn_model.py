'''
We build a simple model from zero to predict the outcome of a rock-paper-scissors game using python.
Using only SoftMax
- Input: 300x200 (RGB) images of hand gestures
- Output: 3 classes (rock, paper, scissors)

Steps:
1. Load the dataset, turn the images to grey, then seperate to train and test sets (80%, 20%)
2. Create a model
3. Train the model
4. Test the model

---Dataset: https://www.kaggle.com/drgfreeman/rockpaperscissors---
DESCRIPTION: This dataset contains images of hand gestures from the Rock-Paper-Scissors game. 
CONTENTS: The dataset contains a total of 2188 images corresponding to:
'Rock' (726 images)
'Paper' (710 images)
'Scissors' (752 images)
All image are taken on a green background with relatively consistent ligithing and white balance.
FORMAT: All images are RGB images of 300 pixels wide by 200 pixels high in .png format. 

The images are separated in three sub-folders named 'rock', 'paper' and 'scissors', according to their respective class.
# images are located in dataset folder. for example: dataset/rock contains 726 images of rock hand gesture.
'''

# Import libraries
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch
import cv2
from rps_nn_model import NNModel
import matplotlib.pyplot as plt
# define seed
seed = 12

################# Loading the dataset #################
# Path to paper, rock, scissors folders
rock_path = 'dataset/rock'
paper_path = 'dataset/paper'
scissors_path = 'dataset/scissors'
# Get the list of images in each folder
rock_images = os.listdir(rock_path)
paper_images = os.listdir(paper_path)
scissors_images = os.listdir(scissors_path)
'''
Here we read the photos to a numpy array of shape (300, 200) each, after greyscaling.
We will use the following labels:
0: rock     1: paper     2: scissors
'''


def read_images_and_labels(path, label):
    images = []
    labels = []
    for image in os.listdir(path):
        # Read the image
        img = cv2.imread(os.path.join(path, image))
        # Resize the image to (300, 200)
        img = cv2.resize(img, (300, 200))
        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Append the image to the images list
        images.append(img)
        # Append the label to the labels list
        labels.append(label)
    # Convert the images and labels lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Read the rock, paper and scissors images and labels
rock_images, rock_labels = read_images_and_labels(rock_path, 0)
paper_images, paper_labels = read_images_and_labels(paper_path, 1)
scissors_images, scissors_labels = read_images_and_labels(scissors_path, 2)

# Concatenate the rock, paper and scissors images and labels
images = np.concatenate((rock_images, paper_images, scissors_images))
labels = np.concatenate((rock_labels, paper_labels, scissors_labels))
# Normalize pixel values
images = images/255.0

test_size = 0.2
# Here we split the data to train and test sets (80%, 20%), using train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=test_size, random_state=seed)

################# Creating the model #################
model = NNModel()
loss_function = torch.nn.CrossEntropyLoss()

################# Training #################
# Now we train the model
# We plot later the loss, validation loss and validation accuracy
losses = []
# Define the number of epochs
epochs = 100
# Define the batch size
batch_size = 32
learning_rate = 0.001
# Loop through the epochs
for epoch in range(epochs):
    # Define the total loss
    total_loss = 0
    # Loop through the training images
    for i in range(0, len(train_images), batch_size):
        # Get the batch images
        batch_images = train_images[i:i+batch_size]
        # Get the batch labels
        batch_labels = train_labels[i:i+batch_size]
        # Convert the batch images to tensor
        batch_images = torch.tensor(batch_images, dtype=torch.float32)
        # Convert the batch labels to tensor
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        # Get the model predictions
        predictions = model(batch_images)
        # Calculate the loss
        loss = loss_function(predictions, batch_labels)
        # Zero the gradients
        model.zero_grad()
        # Backpropagate the loss
        loss.backward()
        # Update the weights using the optimizer
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
        # Add the loss to the total loss
        total_loss += loss.item()
    # add the loss to the losses list
    losses.append(total_loss)
    # Print the loss
    print(f'Epoch: {epoch}, Loss: {total_loss}')


################# Testing #################
# Here we loop through the test set, print the accuracy and show the loss plot.
correct_predictions = 0
model.eval()
for i in range(len(test_images)):
    # Get the image
    image = test_images[i]
    # Get the label
    label = test_labels[i]
    # Convert the image to tensor
    image = torch.tensor(image, dtype=torch.float32)
    # Get the model prediction
    prediction = model(image)
    # Get the predicted label
    predicted_label = torch.argmax(prediction)
    if predicted_label.item() == label:
        # Increment the number of correct predictions
        correct_predictions += 1
accuracy = correct_predictions/len(test_images)
print(f'Accuracy: {accuracy}')

plt.plot(losses)
plt.title('Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

################# Saving the model #################
# save the model state dictionary, so we can load it later
torch.save(model.state_dict(), 'rps_nn_model.pt')
