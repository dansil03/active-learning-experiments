import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy import stats


class OriginalBADGE:
    def __init__(self, model_class, trainset, testloader, device, criterion, optimizer_fn, batch_size):
        self.model_class = model_class
        self.trainset = trainset
        self.testloader = testloader
        self.device = device
        self.criterion = criterion
        self.optimizer_fn = optimizer_fn
        self.batch_size = batch_size

        self.model = self.model_class().to(device)
        self.optimizer = self.optimizer_fn(self.model)

        self.labeled_indices = []
        self.unlabeled_indices = []
        self.label_counts = []
        self.accuracies = []

    def get_dataloaders(self):
        labeled_set = Subset(self.trainset, self.labeled_indices)
        unlabeled_set = Subset(self.trainset, self.unlabeled_indices)
        trainloader = DataLoader(labeled_set, batch_size=self.batch_size, shuffle=True)
        unlabel_loader = DataLoader(unlabeled_set, batch_size=self.batch_size, shuffle=False)
        return trainloader, unlabel_loader

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

    def extract_embeddings_and_probs(self, dataloader):
        self.model.eval()
        embs, probs = [], []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                logits = self.model(x)
                prob = F.softmax(logits, dim=1).cpu()
                feat = self.model.avgpool(
                    self.model.layer4(
                        self.model.layer3(
                            self.model.layer2(
                                self.model.layer1(
                                    self.model.relu(
                                        self.model.bn1(
                                            self.model.conv1(x)
                                        )))))))
                feat = feat.view(feat.size(0), -1).cpu()
                embs.append(feat)
                probs.append(prob)
        return torch.cat(embs), torch.cat(probs)

    def init_centers(self, X1, X2, chosen, chosen_list, mu, D2):
        if len(chosen) == 0:
            ind = torch.argmax(X1[1] * X2[1]).item()
            mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
            D2 = self.distance(X1, X2, mu[0])
            D2[ind] = 0
        else:
            newD = self.distance(X1, X2, mu[-1])
            D2 = torch.min(D2, newD)
            D2[chosen_list] = 0
            probs = (D2 ** 2) / torch.sum(D2 ** 2)
            ind = torch.multinomial(probs, 1).item()
            while ind in chosen:
                ind = torch.multinomial(probs, 1).item()
            mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
        chosen.add(ind)
        chosen_list.append(ind)
        return chosen, chosen_list, mu, D2

    def distance(self, X1, X2, mu):
        X1_vec, X1_norm_sq = X1
        X2_vec, X2_norm_sq = X2
        Y1_vec, Y1_norm_sq = mu[0]
        Y2_vec, Y2_norm_sq = mu[1]
        dists = X1_norm_sq * X2_norm_sq + Y1_norm_sq * Y2_norm_sq - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec)
        return torch.sqrt(torch.clamp(dists, min=0.0))

    def query(self, n):
        _, unlabel_loader = self.get_dataloaders()
        embs, probs = self.extract_embeddings_and_probs(unlabel_loader)
        torch.save({
            "embeddings": embs,
            "pseudo_labels": torch.argmax(probs, dim=1)
        }, "saved_embeddings_with_labels.pt")
        m = len(embs)
        chosen = set()
        chosen_list = []
        emb_norm_sq = torch.sum(embs ** 2, dim=1)
        max_inds = torch.argmax(probs, dim=1)
        probs = -1 * probs
        probs[torch.arange(m), max_inds] += 1
        prob_norm_sq = torch.sum(probs ** 2, dim=1)
        mu = None
        D2 = None
        for _ in range(n):
            chosen, chosen_list, mu, D2 = self.init_centers((probs, prob_norm_sq), (embs, emb_norm_sq), chosen, chosen_list, mu, D2)

        if len(chosen_list) >= 4:
            torch.save({
                "embeddings": embs.cpu(),
                "pseudo_labels": torch.argmax(probs, dim=1).cpu(),
                "selected_indices": chosen_list[:4]
            }, f"kmeans_plus_plus_steps_{len(self.labeled_indices)}.pt")

            print("Gekozen indices opgeslagen:", chosen_list[:4])

        print("Aantal gradient embeddings:", len(embs))
        print("Probs shape:", probs.shape)
        print("Aantal gekozen centers (tijdens query):", len(chosen_list))


        return [self.unlabeled_indices[i] for i in chosen_list]
    


    def run(self, num_iterations=10, query_size=100, reset_model_each_round=True, epochs_per_round=5):
        self.iteration_times = []
        import time
        start_total = time.time()
        for i in range(num_iterations):
            t0 = time.time()
            trainloader, _ = self.get_dataloaders()
            if reset_model_each_round:
                self.model = self.model_class().to(self.device)
                self.optimizer = self.optimizer_fn(self.model)
            self.train(trainloader, epochs_per_round)
            selected = self.query(query_size)
            self.labeled_indices.extend(selected)
            self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in selected]
            acc = self.evaluate()
            self.accuracies.append(acc)
            self.label_counts.append(len(self.labeled_indices))
            self.iteration_times.append(time.time() - t0)
        self.total_time = time.time() - start_total

