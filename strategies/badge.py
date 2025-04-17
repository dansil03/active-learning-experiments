import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

class BADGE:
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
        self.label_counts = []
        self.accuracies = []

    def initialize_dataset(self, initial_size=1000):
        pass

    def get_dataloaders(self, unlabeled_subset_size=None):
        labeled_set = Subset(self.trainset, self.labeled_indices)
        if unlabeled_subset_size and len(self.unlabeled_indices) > unlabeled_subset_size:
            sampled_unlabeled = random.sample(self.unlabeled_indices, unlabeled_subset_size)
        else:
            sampled_unlabeled = self.unlabeled_indices
        trainloader = DataLoader(labeled_set, batch_size=self.batch_size, shuffle=True)
        unlabel_loader = DataLoader(Subset(self.trainset, sampled_unlabeled), batch_size=self.batch_size, shuffle=False)
        return trainloader, unlabel_loader, sampled_unlabeled


    def train(self, trainloader, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_samples = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
            avg_loss = total_loss / total_samples
            print(f"[TRAIN] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


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
    

    def extract_combined_embeddings(self, dataloader):
        self.model.eval() # model in evaluatie modus 
        grad_embeddings = [] 
        norms = []  

        for inputs, _ in dataloader: # datapunten uit de pool hebben een (x, y) structuur waarbij y (het label) nog onbekend is. 
            inputs = inputs.to(self.device) # cuda
            outputs = self.model(inputs) # Hypotetische labels voorspellen

            # Feature extraction
            features = self.model.avgpool(  # downsampling door het onderverdelen van de input naar pooling regions en het gemiddelde waarden van iedere region te berekenen. 
                self.model.layer4(
                    self.model.layer3(
                        self.model.layer2(
                            self.model.layer1(
                                self.model.relu(
                                    self.model.bn1(
                                        self.model.conv1(inputs)
                                    )
                                )
                            )
                        )
                    )
                )
            )
            features = features.view(features.size(0), -1)
            features = F.normalize(features, dim=1)  # Normalize features

            probs = F.softmax(outputs.detach(), dim=1)
            preds = probs.argmax(dim=1)
            probs[range(len(preds)), preds] -= 1
            probs = F.normalize(probs, dim=1)  # Normalize softmax deltas

            B, K = probs.shape
            gx = torch.bmm(probs.unsqueeze(2), features.unsqueeze(1))  # (B, K, d)
            gx = gx.view(B, -1)  # (B, K*d)

            norms.append(gx.norm(dim=1).mean().item())
            grad_embeddings.append(gx.detach().cpu())

        embeddings = torch.cat(grad_embeddings, dim=0)

        print("[DEBUG] Extracting gradient embeddings...")
        print(f"[DEBUG] Shape: {gx.shape}")
        print(f"[DEBUG] Mean grad norm: {gx.norm(dim=1).mean().item():.2f}")
        print(f"[DEBUG] Mean feature norm: {features.norm(dim=1).mean().item():.2f}")
        print(f"[DEBUG] Mean softmax delta norm: {probs.norm(dim=1).mean().item():.2f}")


        try:
            print("[DEBUG] Performing PCA on gradients...")
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings.numpy())
            plt.scatter(reduced[:, 0], reduced[:, 1], s=5)
            plt.title("Gradient Embeddings (PCA)")
            plt.show()
        except Exception as e:
            print(f"[DEBUG] PCA skipped: {e}")

        print(f"[DEBUG] Mean norm after normalization: {sum(norms)/len(norms):.2f}")
        return embeddings

    def kmeanspp_torch(self, data, k):
        n_samples = data.size(0)
        centers = [data[torch.randint(0, n_samples, (1,), device=self.device).item()]]
        for _ in range(1, k):
            center_stack = torch.stack(centers)
            dists = torch.cdist(data, center_stack).min(dim=1).values
            probs = dists ** 2
            probs /= probs.sum()
            next_index = torch.multinomial(probs, 1).item()
            centers.append(data[next_index])
        return torch.stack(centers)

    def query(self, data, k):
        data_tensor = data.to(self.device) if not isinstance(data, torch.Tensor) else data
        centers = self.kmeanspp_torch(data_tensor, k)
        selected_indices = []
        for center in centers:
            dists = torch.norm(data_tensor - center, dim=1)
            selected_indices.append(torch.argmin(dists).item())
        return selected_indices

    def run(self, num_iterations=15, query_size=100, embed_limit=10000, unlabeled_subset_size=30000, reset_model_each_round=True, epochs_per_round=10):
        self.iteration_times = []
        start_total = time.time()
        for round in range(num_iterations):
            iter_start = time.time()

            trainloader, unlabel_loader, sampled_unlabeled = self.get_dataloaders(unlabeled_subset_size)

            if reset_model_each_round:
                self.model = self.model_class().to(self.device)
                self.optimizer = self.optimizer_fn(self.model)

            self.train(trainloader, epochs=epochs_per_round)

            combined = self.extract_combined_embeddings(unlabel_loader)

            selected_relative = self.query(combined, query_size)
            selected_absolute = [sampled_unlabeled[i] for i in selected_relative]

            self.labeled_indices.extend(selected_absolute)
            self.unlabeled_indices = [i for i in self.unlabeled_indices if i not in selected_absolute]

            acc = self.evaluate()
            self.accuracies.append(acc)
            self.label_counts.append(len(self.labeled_indices))

            iter_end = time.time()
            self.iteration_times.append(iter_end - iter_start)

        self.total_time = time.time() - start_total
