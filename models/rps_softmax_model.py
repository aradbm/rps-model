import torch
seed = 12


class SimpleSoftmaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # A linear layer
        self.layer = torch.nn.Linear(300*200, 3)
        # Softmax activation
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 300*200)
        x = self.layer(x)
        return self.softmax(x)

    def init_weights(self):
        torch.manual_seed(seed)
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.zeros_(self.layer.bias)
