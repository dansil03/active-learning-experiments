from .badge import BADGE
from .random_sampler import RandomSampler
from .batchbald.strategy import BatchBALDStrategy


def get_strategy(name):
    name = name.lower()
    if name == "badge":
        return BADGE
    elif name == "random":
        return RandomSampler
    elif name == "batchbald": 
        return BatchBALDStrategy
    else:
        raise ValueError(f" Unsupported strategy: {name}")
