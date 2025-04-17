import yaml
import mlflow
import time
import matplotlib.pyplot as plt
from models import get_model
from strategies import get_strategy
from datasets import get_dataset 
from torch.utils.data import DataLoader
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
from utils.tracking import log_metrics, log_params, log_duration, log_plot
from utils.seed import seed_everything
import multiprocessing as mp  
mp.set_start_method("spawn", force=True)

from multiprocessing import Process
import copy


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_strategy(cfg, run_idx): 
    model_name = cfg["model"]
    strategy_name = cfg["strategy"]
    dataset_name = cfg["dataset"]
    initial_size = cfg["initial_size"]
    query_sizes = cfg["query_size"] if isinstance(cfg["query_size"], list) else [cfg["query_size"]]

    trainset, testset = get_dataset(dataset_name)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    for query_size in query_sizes:
        # Zet unieke seed
        seed = 42 + run_idx
        seed_everything(seed)

        model_fn = get_model(model_name)
        model = model_fn()

        optimizer_fn = lambda model: optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        criterion = CrossEntropyLoss()

        strategy_cls = get_strategy(strategy_name)
        learner = strategy_cls(
            model_class=model_fn,
            trainset=trainset,
            testloader=testloader,
            device=device,
            criterion=criterion,
            optimizer_fn=optimizer_fn,
            batch_size=32
        )

        run_name = f"{strategy_name}_{model_name}_{dataset_name}_init{initial_size}_q{query_size}_run{run_idx}"

        with mlflow.start_run(run_name=run_name):
            log_params({
                "model": model_name,
                "strategy": strategy_name,
                "dataset": dataset_name,
                "initial_size": initial_size,
                "query_size": query_size,
                "seed": seed
            })

            # Init labeled/unlabeled indices
            all_indices = list(range(len(trainset)))
            learner.labeled_indices = sorted(all_indices[:initial_size])
            learner.unlabeled_indices = sorted(list(set(all_indices) - set(learner.labeled_indices)))

            i = 0
            while len(learner.unlabeled_indices) >= query_size:
                iter_start = time.time()

                learner.run(
                    num_iterations=1,
                    query_size=query_size,
                    epochs_per_round=5,
                    reset_model_each_round=True
                )

                acc = learner.accuracies[-1]
                n_labels = learner.label_counts[-1]
                log_metrics({"accuracy": acc, "label_count": n_labels}, step=i)

                log_duration(iter_start, label=f"iteration_time_{i}")
                i += 1

            mlflow.log_metric("total_time", learner.total_time)

            # Plot
            fig, ax = plt.subplots()
            ax.plot(learner.label_counts, learner.accuracies)
            ax.set_xlabel("Labeled examples")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy vs. Labeled examples")
            log_plot(fig, name="accuracy_curve")

            print(f" Finished: {run_name}")


if __name__ == "__main__":
    config = load_config()
    REPEATS = 3
    processes = []

    for cfg in config["experiments"]:
        for run_idx in range(REPEATS):
            cfg_copy = copy.deepcopy(cfg)
            p = Process(target=run_strategy, args=(cfg_copy, run_idx))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
