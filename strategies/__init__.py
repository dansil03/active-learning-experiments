from .badge import BADGE
from .random_sampler import RandomSampler

def get_strategy(name):
    name = name.lower()
    if name == "badge":
        return BADGE
    elif name == "random":
        return RandomSampler
    else:
        raise ValueError(f" Unsupported strategy: {name}")
