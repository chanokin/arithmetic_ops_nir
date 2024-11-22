import torch


# Create a minimal network
class AdditionNetwork(torch.nn.Module):
    def __init__(self, n_input=3):
        super().__init__()

        self.module_0 = torch.nn.Linear(n_input, 1)
        self.module_1 = torch.nn.Linear(n_input, 1)
        self.output = torch.nn.ReLU()

    def forward(self, x):
        x0 = self.input_0(x)
        x1 = self.input_1(x)
        x = x0 + x1
        return self.output(x)


class AdditionModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class HackAdditionNetwork(torch.nn.Module):
    def __init__(self, n_input=3):
        super().__init__()

        self.module_0 = torch.nn.Linear(n_input, 1)
        self.module_1 = torch.nn.Linear(n_input, 1)
        self.output = torch.nn.ReLU()
        self.add_m0_m1 = AdditionModule()

    def forward(self, x):
        x0 = self.input_0(x)
        x1 = self.input_1(x)
        x3 = self.add_m0_m1(x0, x1)
        return self.output(x3)
