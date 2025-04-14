# Active Learning Experimental research



## Research Question

***Which Active Learning methods result in the greatest reduction in required labels without significant performance degradation?***

---

## Hypotheses

### Null Hypotheses

- **H0.1**: There is no significant difference in accuracy between models trained with BADGE and Random Sampling, given the same number of labeled datapoints.
- **H0.2**: The time efficiency of BADGE is not significantly lower than that of Random Sampling.

---

## Experiment Setup

### Experiment v1.0 --> BADGE vs. Random Sampling 
| Parameter       | Value(s)                                 |
|----------------|-------------------------------------------|
| Dataset         | CIFAR-10                                 |
| Models          | ResNet18                                 |
| Strategies      | BADGE, Random Sampling                   |
| Initial Sizes   | 5000                                     |
| Query Sizes     | 500                                      |
| Iterations      | 150                                      |
| Evaluation Tool | MLflow                                   |
| Logging         | Accuracy, Time per Iteration, Total Time |
| Environment     | CUDA-enabled VM, Python 3.10, PyTorch 2  |

---