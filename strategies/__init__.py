from .badge import BADGE
from .original_badge import OriginalBADGE
from .random_sampler import RandomSampler
from .batchbald.strategy import BatchBALDStrategy
from .uncertainty_sampling import UncertaintySampler


def get_strategy(name):
    name = name.lower()
    if name == "badge":
        return BADGE
    elif name == "original_badge": 
        return OriginalBADGE
    elif  name == "badge_test": 
        return OriginalBADGE
    elif name == "random":
        return RandomSampler
    elif name == "random_test": 
        return RandomSampler
    elif name == "batchbald": 
        return BatchBALDStrategy
    elif name == "uncertainty": 
        return UncertaintySampler
    else:
        raise ValueError(f" Unsupported strategy: {name}")
