import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch
import torch
from rps_simple_model import MyModel

# Initialize the model
model = MyModel()

# Load the state dictionary
state_dict = torch.load('rps_model.pt')
model.load_state_dict(state_dict)

# # now we want to test the model on a random image
# # read the image
# img = cv2.imread('test_image.jpg')
# # resize the image to (300, 200)
# img = cv2.resize(img, (300, 200))
# # convert the image to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # convert the image to a numpy array
# img = np.array(img)
# # convert the image to a tensor
# img = torch.from_numpy(img)
# # add a batch dimension
# img = img.unsqueeze(0)
# # predict the class of the image
