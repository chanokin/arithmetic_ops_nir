import norse
from modules.minimal_net import AdditionNetwork
import nirtorch as nt
import torch


def convert_to_nir():
    net = AdditionNetwork()
    norse.to_nir(net, "addition_nir")


def get_graph():
    net = AdditionNetwork()
    sample = torch.randn(1, 1)
    nt.extract_torch_graph(net, sample, "addition_nir")


if __name__ == "__main__":
    convert_to_nir()
