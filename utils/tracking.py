import mlflow
import time
import matplotlib.pyplot as plt
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("test experiment")


def log_params(params):
    """Log parameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics, step=None):
    """Log metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)

def log_plot(fig, name="plot"):
    """Log a matplotlib figure to MLflow."""
    plot_path = f"{name}.png"
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close(fig)
    os.remove(plot_path)

def log_duration(start_time, label="duration"):
    """Log elapsed time as a metric."""
    duration = time.time() - start_time
    mlflow.log_metric(label, duration)