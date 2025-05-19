import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
import time

class UncertaintySampler:
    def __init__(self, model_class, trainset, testloader, device, criterion, optimizer_fn, batch_size):
        self.model_class = model_class
        self.optimizer_fn = optimizer_fn

        self.model = self.model_class().to(device)
        self.optimizer = self.optimizer_fn(self.model)

        self.trainset = trainset
        self.testloader = testloader
        self.device = device
        self.criterion = criterion
        self.batch_size = batch_size
        self.labeled_indices = []
        self.unlabeled_indices = []
        self.accuracies = []
        self.label_counts = []

    def initialize_dataset(self, initial_size=500):
        all_indices = list(range(len(self.trainset)))
        self.labeled_indices = random.sample(all_indices, initial_size)
        self.unlabeled_indices = list(set(all_indices) - set(self.labeled_indices))

    def get_dataloaders(self):
        labeled_set = Subset(self.trainset, self.labeled_indices)
        trainloader = DataLoader(labeled_set, batch_size=self.batch_size, shuffle=True)
        return trainloader

    def train(self, trainloader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in self.testloader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
        return 100 * correct / total

    def query(self, query_size):
        self.model.eval()
        subset = Subset(self.trainset, self.unlabeled_indices)
        loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)

        all_margins = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                top2 = torch.topk(probs, 2, dim=1).values
                margin = top2[:, 0] - top2[:, 1]
                all_margins.append(margin.cpu())

        margins = torch.cat(all_margins)  
        indices = torch.argsort(margins)[:query_size]  
        selected = [self.unlabeled_indices[i.item()] for i in indices]
        return selected

    def run(self, num_iterations=5, query_size=10, epochs_per_round=1, reset_model_each_round=True):
        if not self.labeled_indices or not self.unlabeled_indices:
            self.initialize_dataset()

        self.iteration_times = []
        start_total = time.time()

        for i in range(num_iterations):
            if reset_model_each_round:
                self.model = self.model_class().to(self.device)
                self.optimizer = self.optimizer_fn(self.model)

            trainloader = self.get_dataloaders()
            self.train(trainloader, epochs=epochs_per_round)

            acc = self.evaluate()
            self.accuracies.append(acc)

            newly_selected = self.query(query_size)
            self.labeled_indices.extend(newly_selected)
            self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in newly_selected]

            self.label_counts.append(len(self.labeled_indices))

        self.total_time = time.time() - start_total
