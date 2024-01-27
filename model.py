import torch
seed = 12


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        hidden_size = 5
        # Define the first layer
        self.layer1 = torch.nn.Linear(300*200, hidden_size)
        # Define the second layer
        self.layer2 = torch.nn.Linear(hidden_size, 3)
        # Define the activation function
        self.activation = torch.nn.ReLU()
        # Define the softmax function
        self.softmax = torch.nn.Softmax(dim=1)
        # Initialize the weights
        self.init_weights()

    def forward(self, x):
        x = x.view(-1, 300*200)  # vectorize the input to shape (1, 300*200)
        # Pass the input through the first layer
        x = self.layer1(x)
        # Pass the output through relu activation function
        x = self.activation(x)
        # Pass the output through the second layer
        x = self.layer2(x)
        # Return the output through softmax
        return self.softmax(x)

    def init_weights(self):
        # use seed to make sure the weights are initialized the same way every time
        torch.manual_seed(seed)
        # Initialize the weights of the first layer
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        # Initialize the weights of the second layer
        torch.nn.init.xavier_uniform_(self.layer2.weight)
