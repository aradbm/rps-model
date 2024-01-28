import torch
seed = 12


class NNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 5
        self.layer1 = torch.nn.Linear(300*200, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, 3)
        self.activation = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, x):
        x = x.view(-1, 300*200)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return self.softmax(x)

    def init_weights(self):
        torch.manual_seed(seed)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
