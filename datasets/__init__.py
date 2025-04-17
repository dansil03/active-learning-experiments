from torchvision.datasets import CIFAR10, MNIST
import torchvision.transforms as T

def get_dataset(name):
    name = name.lower()
    transform = T.Compose([T.ToTensor()])

    if name == "cifar10":
        trainset = CIFAR10(root="./datasets/cifar10", train=True, download=True, transform=transform)
        testset = CIFAR10(root="./datasets/cifar10", train=False, download=True, transform=transform)
    elif name == "mnist":
        trainset = MNIST(root="./datasets/mnist", train=True, download=True, transform=transform)
        testset = MNIST(root="./datasets/mnist", train=False, download=True, transform=transform)
    else:
        raise ValueError(f" Unsupported dataset: {name}")

    return trainset, testset
