import os
import cv2
import torch
from rps_nn_model import NNModel
import matplotlib.pyplot as plt
import tkinter as tk

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
grid_size = 3 * 5  # 3 rows, 5 columns
current_image = 1  # Start with 1

###### Load the model ######
model = NNModel()
model.load_state_dict(torch.load('rps_nn_model.pt'))

###### Load the images ######
rock_path = 'dataset/rock'
paper_path = 'dataset/paper'
scissors_path = 'dataset/scissors'
rock_images = os.listdir(rock_path)
paper_images = os.listdir(paper_path)
scissors_images = os.listdir(scissors_path)


def show_images(images, labels, path):
    global current_image
    for i in range(5):
        img_path = os.path.join(path, images[i])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (300, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0

        pred = model(torch.tensor(img).float().unsqueeze(0).view(-1, 300*200))
        pred = torch.argmax(pred).item()

        ax = fig.add_subplot(3, 5, current_image)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Predicted: {pred}, True: {labels[i]}')
        ax.set_xticks([])
        ax.set_yticks([])

        current_image += 1


###### Show the images ######
show_images(rock_images, [0]*5, rock_path)
show_images(paper_images, [1]*5, paper_path)
show_images(scissors_images, [2]*5, scissors_path)

# Show the figure
plt.show()
