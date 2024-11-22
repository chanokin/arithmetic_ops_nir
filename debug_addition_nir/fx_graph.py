import torch
from torch.fx import symbolic_trace
import norse


# Create a simple model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.li = norse.torch.LICell()

    def forward(self, x):
        x = self.conv(x)
        x = self.li(x)
        return x


def test_fx_trace():
    model = Model()
    x = torch.rand(1, 1)

    traced = symbolic_trace(model)
    print(traced.graph)


if __name__ == "__main__":
    test_fx_trace()
