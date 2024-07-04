import os
import cv2
import torch
from rps_cnn_model import CNNModel
import matplotlib.pyplot as plt
import tkinter as tk
import random


num_images_per_class = 8

###### Initialize Tkinter and get screen dimensions ######
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()
# Set figure size to be 3/4 of the screen size
fig_width = screen_width * 0.75 / 100
fig_height = screen_height * 0.75 / 100
# Initialize the figure with the new size
fig = plt.figure(figsize=(fig_width, fig_height))
grid_size = 3 * num_images_per_class  # rows, columns
current_image = 1  # Start with 1

###### Load the model ######
model = CNNModel()
model.load_state_dict(torch.load('rps_cnn_model.pt',
                      map_location=torch.device('cpu')))


###### Load the images ######
rock_path = 'dataset/rock'
paper_path = 'dataset/paper'
scissors_path = 'dataset/scissors'

rock_images = random.sample(os.listdir(rock_path), min(
    num_images_per_class, len(os.listdir(rock_path))))
paper_images = random.sample(os.listdir(paper_path), min(
    num_images_per_class, len(os.listdir(paper_path))))
scissors_images = random.sample(os.listdir(scissors_path), min(
    num_images_per_class, len(os.listdir(scissors_path))))
model.eval()


def show_images(images, labels, path):
    global current_image
    # Show up to num_images_per_class images
    for i in range(num_images_per_class):
        img = cv2.imread(os.path.join(path, images[i]))
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = torch.tensor(img).float()
        # normalize the image
        img = img / 255.0
        # Predict the image
        pred = model(img.unsqueeze(0)).argmax().item()
        true = labels[i]
        # Add the subplot to the figure
        ax = fig.add_subplot(3, num_images_per_class, current_image)
        # Remove the x and y ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(img)
        ax.set_title('Pred: ' + str(pred) + ' True: ' + str(true))
        current_image += 1


###### Show the images ######
show_images(rock_images, [0] * num_images_per_class, rock_path)
show_images(paper_images, [1] * num_images_per_class, paper_path)
show_images(scissors_images, [2] * num_images_per_class, scissors_path)

# Show the figure
plt.show()
