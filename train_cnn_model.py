'''
We build a simple model using CNN
to predict the outcome of a rock-paper-scissors game using python.

- Input: 300x200 (grey) images of hand gestures
- Output: 3 classes (rock, paper, scissors)

Steps:
1. Load the dataset 
2. Turn the images to grey & seperate to train and test sets (80%, 20%)
2. Create a model
3. Train the model
4. Test the model

We will use the following labels:
0: rock     1: paper     2: scissors
'''
import os
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rps_cnn_model import CNNModel
from sklearn.model_selection import train_test_split
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


def read_images_and_labels(path, label, augment=False):
    images = []
    labels = []
    for image in os.listdir(path):
        # Read and preprocess the image
        img = cv2.imread(os.path.join(path, image))
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Append the original image to the dataset
        images.append(img)
        labels.append(label)

        if augment:
            # Apply augmentation
            augmented_img = img.copy()

            # Rotation
            angle = random.randint(-15, 15)
            M = cv2.getRotationMatrix2D((150, 100), angle, 1)
            augmented_img = cv2.warpAffine(augmented_img, M, (300, 200))

            # Scaling
            scale = random.uniform(0.9, 1.1)
            augmented_img = cv2.resize(
                augmented_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            augmented_img = cv2.resize(augmented_img, (300, 200))

            # Adding Noise
            gauss = np.random.normal(0, 1, augmented_img.size)
            gauss = gauss.reshape(
                augmented_img.shape[0], augmented_img.shape[1]).astype('uint8')
            augmented_img = cv2.add(augmented_img, gauss)

            # Append the augmented image to the dataset
            images.append(augmented_img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


# Read the rock, paper and scissors images and labels
rock_images, rock_labels = read_images_and_labels(rock_path, 0, augment=True)
paper_images, paper_labels = read_images_and_labels(
    paper_path, 1, augment=True)
scissors_images, scissors_labels = read_images_and_labels(
    scissors_path, 2, augment=True)

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
model = CNNModel()
loss_function = torch.nn.CrossEntropyLoss()

################# Training #################
# Now we train the model
# We plot later the loss, validation loss and validation accuracy
losses = []
accuracies = []
# Define the number of epochs
epochs = 2
learning_rate = 0.001
# Define the batch size
batch_size = 32
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Convert the train and test images to torch compatible format
train_images = torch.from_numpy(train_images).float()
test_images = torch.from_numpy(test_images).float()
# Convert the train and test labels to torch compatible format
train_labels = torch.from_numpy(train_labels).long()
test_labels = torch.from_numpy(test_labels).long()

# Loop through the epochs
for epoch in range(epochs):
    # Shuffle the train images and labels
    permutation = torch.randperm(train_images.shape[0])
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]
    # Loop through the train images
    for i in range(0, train_images.shape[0], batch_size):
        # Get the images and labels for this iteration
        images = train_images[i:i+batch_size]
        labels = train_labels[i:i+batch_size]
        # Zero out the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images.unsqueeze(1))
        # Calculate the loss
        loss = loss_function(outputs, labels)
        # Backward pass
        loss.backward()
        # Update the weights
        optimizer.step()
    ################# Validation #################
    # Calculate the loss on the test set
    test_outputs = model(test_images.unsqueeze(1))
    test_loss = loss_function(test_outputs, test_labels)
    # Calculate the accuracy on the test set
    _, predicted = torch.max(test_outputs.data, 1)
    correct_predictions = (predicted == test_labels).sum().item()
    accuracy = correct_predictions / test_labels.size(0)
    print('Epoch: {}, Loss: {}, Accuracy: {}'.format(
        epoch, test_loss.item(), accuracy))
    accuracies.append(accuracy)
    losses.append(loss.item())


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
    prediction = model(image.unsqueeze(0).unsqueeze(0))
    # Get the predicted label
    predicted_label = torch.argmax(prediction)
    if predicted_label.item() == label:
        # Increment the number of correct predictions
        correct_predictions += 1
accuracy = correct_predictions/len(test_images)
print(f'Final model accuracy: {accuracy}')

# plot the loss and the accuracy
plt.plot(losses)
plt.plot(accuracies)
plt.title('Loss and Accuracy of the model')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(['Loss', 'Accuracy'])
plt.show()

################# Saving the model #################
# save the model state dictionary, so we can load it later
torch.save(model.state_dict(), 'rps_cnn_model.pt')
