import norse
from modules.minimal_net import AdditionNetwork


def convert_to_nir():
    net = AdditionNetwork()
    norse.to_nir(net, "addition_nir")


if __name__ == "__main__":
    convert_to_nir()
