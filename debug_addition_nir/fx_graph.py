from collections import OrderedDict
import torch
from torch.fx import Tracer, GraphModule
import norse

"""
from: https://github.com/pytorch/pytorch/issues/51803
"""


class CustomisableTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """

    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True

        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
            return True

        return m.__module__.startswith("torch.nn") and not isinstance(m, torch.nn.Sequential)


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


class AdditionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.li = norse.torch.LICell()

    def forward(self, x):
        x = self.conv(x)
        y = self.li(x)
        z = x + y
        return z


def test_fx_trace():
    model = Model()
    tracer = CustomisableTracer(customed_leaf_module=(norse.torch.LICell,))
    tracer_graph = tracer.trace(model)
    traced_model = GraphModule(tracer.root, tracer_graph, "test_fx_network")
    named_modules = OrderedDict(traced_model.named_modules())

    # two internal modules and the model itself
    assert len(named_modules) == 3
    assert "conv" in named_modules
    assert "li" in named_modules

    print(named_modules)


def test_addition_trace():
    model = AdditionModel()
    tracer = CustomisableTracer(customed_leaf_module=(norse.torch.LICell,))
    tracer_graph = tracer.trace(model)
    traced_model = GraphModule(tracer.root, tracer_graph, "test_addition_network")

    named_modules = OrderedDict(traced_model.named_modules())

    # two internal modules and the model itself
    assert len(named_modules) == 3
    assert "conv" in named_modules
    assert "li" in named_modules

    graph = traced_model.graph
    has_addition = False
    for node in graph.nodes:
        if node.op == "call_function" and node.name == "add":
            has_addition = True

            # x in the model; which comes from the conv layer
            assert node.args[0].name == "conv"
            # y in the model; which comes from the LICell layer
            assert node.args[1].name == "li"

            break
    assert has_addition


if __name__ == "__main__":
    test_fx_trace()
    test_addition_trace()
