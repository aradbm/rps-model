import torch
seed = 12


class SimpleSoftmaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a single layer that directly connects the input to the 3 output classes
        self.layer = torch.nn.Linear(300*200, 3)
        # Define the softmax function
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 300*200)  # Vectorize the input to shape (1, 300*200)
        # Pass the input through the layer
        x = self.layer(x)
        # Return the output through softmax
        return self.softmax(x)

    def init_weights(self):
        torch.manual_seed(seed)
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.zeros_(self.layer.bias)
