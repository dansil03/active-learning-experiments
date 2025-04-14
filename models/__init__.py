from .resnet import ResNet18, BasicBlock

def get_model(name):
    if name.lower() == "resnet":
        return lambda: ResNet18(BasicBlock)
    else:
        raise ValueError(f"Model '{name}' not supported.")
