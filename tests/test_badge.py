import torch
import pytest
from strategies.badge import BADGE
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from models import get_model

@pytest.fixture
def dummy_badge():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    model_fn = get_model("resnet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    badge = BADGE(
        model_class=model_fn,
        trainset=dataset,
        testloader=DataLoader(dataset, batch_size=32),
        device=device,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_fn=lambda m: torch.optim.SGD(m.parameters(), lr=0.01),
        batch_size=8
    )
    badge.labeled_indices = list(range(100))  # dummy initialization
    badge.unlabeled_indices = list(range(100, 200))
    return badge

def test_embeddings_shape(dummy_badge):
    _, unlabel_loader, _ = dummy_badge.get_dataloaders()
    emb = dummy_badge.extract_combined_embeddings(unlabel_loader)
    assert emb.dim() == 2
    assert emb.shape[1] > 0

def test_query_selection(dummy_badge):
    _, unlabel_loader, sampled_unlabeled = dummy_badge.get_dataloaders()
    emb = dummy_badge.extract_combined_embeddings(unlabel_loader)
    selected = dummy_badge.query(emb, 10)
    assert isinstance(selected, list)
    assert len(selected) == 10
    assert all(isinstance(i, int) for i in selected)

def test_cuda_usage(dummy_badge):
    model = dummy_badge.model
    assert next(model.parameters()).is_cuda == torch.cuda.is_available()

def test_train_run(dummy_badge):
    trainloader, _, _ = dummy_badge.get_dataloaders()
    dummy_badge.train(trainloader, epochs=1)
    assert dummy_badge.model.training is True

def test_evaluate(dummy_badge):
    acc = dummy_badge.evaluate()
    assert isinstance(acc, float)
    assert 0 <= acc <= 100

def test_query_returns_unique_indices(dummy_badge):
    _, unlabel_loader, _ = dummy_badge.get_dataloaders()
    emb = dummy_badge.extract_combined_embeddings(unlabel_loader)
    selected = dummy_badge.query(emb, 10)
    assert len(selected) == len(set(selected)), "Niet-unieke indices gevonden in query-resultaat"


def test_embeddings_change_after_training(dummy_badge):
    _, unlabel_loader, _ = dummy_badge.get_dataloaders()
    
    emb_before = dummy_badge.extract_combined_embeddings(unlabel_loader).clone()
    
    trainloader, _, _ = dummy_badge.get_dataloaders()
    dummy_badge.train(trainloader, epochs=1)
    
    emb_after = dummy_badge.extract_combined_embeddings(unlabel_loader)
    
    diff = torch.norm(emb_before - emb_after).item()
    assert diff > 1e-3, f"Embeddings lijken niet veranderd na training (verschil: {diff})"
