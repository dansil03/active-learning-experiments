from .resnet import ResNet18, BasicBlock
from .mc_resnet import MCResNet18


def get_model(name):
    if name.lower() == "resnet":
        return lambda: ResNet18(BasicBlock)
    elif name.lower() == "mc_resnet":
        return lambda: MCResNet18()
    else:
        raise ValueError(f"Model '{name}' not supported.")
