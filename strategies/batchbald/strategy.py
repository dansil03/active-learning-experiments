import torch
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader, Subset

from strategies.batchbald.batchbald import get_batchbald_batch

class BatchBALDStrategy:
    def __init__(self, model_class, trainset, testloader, device, criterion, optimizer_fn, batch_size=32):
        self.device = device
        self.trainset = trainset
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.batch_size = batch_size
        self.model_class = model_class

        self.labeled_indices = []
        self.unlabeled_indices = []

        self.accuracies = []
        self.label_counts = []
        self.total_time = 0

        self.num_mc_samples = 25
        self.num_joint_samples = 10000

    def train(self):
        model = self.model_class().to(self.device)
        optimizer = self.optimizer_fn(model)
        train_loader = DataLoader(Subset(self.trainset, self.labeled_indices), batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(model(x), y)
                loss.backward()
                optimizer.step()

        self.model = model

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return 100.0 * correct / total

    def predict_log_probs(self, loader):
        self.model.train()  # MC Dropout actief houden
        all_log_probs = []

        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                batch_probs = []
                for x, _ in loader:
                    x = x.to(self.device)
                    output = self.model(x)
                    log_probs = F.log_softmax(output, dim=1)
                    batch_probs.append(log_probs.cpu())
                all_log_probs.append(torch.cat(batch_probs, dim=0).unsqueeze(1))

        return torch.cat(all_log_probs, dim=1)

    def query(self, n):
        loader = DataLoader(Subset(self.trainset, self.unlabeled_indices), batch_size=self.batch_size)
        log_probs = self.predict_log_probs(loader)
        result = get_batchbald_batch(
            log_probs_N_K_C=log_probs,
            batch_size=n,
            num_samples=self.num_joint_samples,
            dtype=torch.float64,
            device=self.device
        )
        return [self.unlabeled_indices[i] for i in result.indices]

    def run(self, num_iterations=1, query_size=100, epochs_per_round=5, reset_model_each_round=True):
        self.epochs = epochs_per_round

        for _ in range(num_iterations):
            start = time.time()

            if reset_model_each_round or not hasattr(self, 'model'):
                self.train()

            acc = self.evaluate()
            self.accuracies.append(acc)
            self.label_counts.append(len(self.labeled_indices))

            query_indices = self.query(query_size)
            self.labeled_indices += query_indices
            self.unlabeled_indices = list(set(self.unlabeled_indices) - set(query_indices))

            self.total_time += time.time() - start
