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
seed = 42

################# Loading the dataset #################
# Path to paper, rock, scissors folders
rock_path = '/kaggle/input/rockpaperscissors/paper'
paper_path = '/kaggle/input/rockpaperscissors/rock'
scissors_path = '/kaggle/input/rockpaperscissors/scissors'
# Get the list of images in each folder
rock_images = os.listdir(rock_path)
paper_images = os.listdir(paper_path)
scissors_images = os.listdir(scissors_path)


def read_images_and_labels(path, label):
    images = []
    labels = []
    for image in os.listdir(path):
        # Read and preprocess the original image
        img = cv2.imread(os.path.join(path, image))
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def random_rotation(image):
    angle = random.randint(-30, 30)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, rotation_matrix, (cols, rows))


def random_flip(image):
    flip_type = random.choice([-1, 0, 1])  # Horizontal, vertical or both
    return cv2.flip(image, flip_type)


# Read the rock, paper and scissors images and labels
rock_images, rock_labels = read_images_and_labels(rock_path,    0)
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
# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
#############################################
dp_rate = 0.3
model = CNNModel(dropout_rate=dp_rate).to(device)
augment_bool = True
epochs = 50
learning_rate = 0.001
#############################################

loss_function = torch.nn.CrossEntropyLoss()
################# Training #################
# Now we train the model
# We plot later the loss, validation loss and validation accuracy
losses = []
accuracies = []

# Define the batch size
batch_size = 32
# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_images = torch.from_numpy(train_images).float().to(device)
test_images = torch.from_numpy(test_images).float().to(device)
train_labels = torch.from_numpy(train_labels).long().to(device)
test_labels = torch.from_numpy(test_labels).long().to(device)

for epoch in range(epochs):
    # Shuffle the train images and labels
    permutation = torch.randperm(train_images.shape[0])
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    for i in range(0, train_images.shape[0], batch_size):
        augmented_images = []
        augmented_labels = []
        end = min(i + batch_size, train_images.shape[0])
        images = train_images[i:end]
        labels = train_labels[i:end]

        for j in range(images.shape[0]):
            original_image = images[j].cpu().numpy()
            augmented_images.append(torch.tensor(
                original_image, dtype=torch.float32).to(device))
            augmented_labels.append(labels[j])

            if augment_bool:
                for _ in range(8):
                    augmented_image = random_rotation(original_image)
#                     augmented_image = random_flip(augmented_image)
                    augmented_image = cv2.resize(augmented_image, (300, 200))
                    augmented_images.append(torch.tensor(
                        augmented_image, dtype=torch.float32).to(device))
                    augmented_labels.append(labels[j])

        augmented_images = torch.stack(augmented_images)
        augmented_labels = torch.stack(augmented_labels)
        optimizer.zero_grad()
        outputs = model(augmented_images.unsqueeze(1))
        loss = loss_function(outputs, augmented_labels)
        loss.backward()
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
total_params = sum(p.numel() for p in model.parameters())

print("#############################################")
print(f"Total number of parameters: {total_params}")
print(f'Drop out rate: {dp_rate}')
print(f'Augment: {augment_bool}')
print(f'Epochs: {epochs}')
print(f'Learning rate: {learning_rate}')
print(f'--Final model accuracy: {accuracy}---')
print("#############################################")

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
