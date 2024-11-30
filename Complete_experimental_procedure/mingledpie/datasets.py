import random
import numpy as np
import torchvision
import torchvision.transforms as transforms

from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from util import *

def load_dataset(dataset_name, label_groups, train=True):
    if dataset_name == 'mnist':
        return _load_MNIST(label_groups, train)
    elif dataset_name == 'fmnist':
        return _load_FMNIST(label_groups, train)
    elif dataset_name == 'cifar10':
        return _load_CIFAR10(label_groups, train)
    elif dataset_name == 'cifar100':
        return _load_CIFAR100(label_groups, train)
    elif dataset_name == 'purchase100':
        return _load_Purchase100(label_groups, train)
    elif dataset_name == 'texas100':
        return _load_Texas100(label_groups, train)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def setup_datasets(data_seed, dataset_name, k, m, m_test, samples_per_subset, label_groups, disruption=False, isomerism=False):

    np.random.seed(data_seed)

    train_dataset = {}
    (X, y, clusters) = load_dataset(dataset_name, label_groups, train=True)
    train_dataset['data_indices'], train_dataset['cluster_assign'] = \
        _setup_dataset(clusters, k, m, label_groups, disruption, isomerism, train=True)
    X_subsets, y_subsets = simulated_data(X, y, samples_per_subset, label_groups)
    train_dataset['X'] = X
    train_dataset['y'] = y
    train_dataset['X_subsets'] = X_subsets
    train_dataset['y_subsets'] = y_subsets
    # train_dataset['train'] = dataset

    test_dataset = {}
    (X, y, clusters) = load_dataset(dataset_name, label_groups, train=False)
    test_dataset['data_indices'], test_dataset['cluster_assign'] = \
        _setup_dataset(clusters, k, m_test, label_groups, disruption, isomerism, train=False)
    test_dataset['X'] = X
    test_dataset['y'] = y

    return train_dataset, test_dataset
    # self.dataset['test'] = dataset


def _setup_dataset(clusters, k, m, label_groups, disruption=False, isomerism=False, train=True):
    data_indices = []
    cluster_assign = []
    if isomerism:
        group_sizes = [len(group) for group in label_groups]
        total_weight = sum(group_sizes)
        m_per_cluster = [int((size / total_weight) * m) for size in group_sizes]

        for k_i in range(k):

            ll2 = chunkify(clusters[k_i], m_per_cluster[k_i]) # splits ll into m lists with size n
            data_indices += ll2
            cluster_assign += [k_i for _ in range(m_per_cluster[k_i])]
    else:
        m_per_cluster = m // k

        for k_i in range(k):
            ll2 = chunkify(clusters[k_i], m_per_cluster)  # splits ll into m lists with size n
            data_indices += ll2
            cluster_assign += [k_i for _ in range(m_per_cluster)]

    if disruption:
        random.shuffle(cluster_assign)
    cluster_assign = np.array(cluster_assign)

    return data_indices, cluster_assign

def simulated_data(X, y, samples_per_subset, label_groups):

    X_subsets = []
    y_subsets = []
    subsets_classes = label_groups

    for classes in subsets_classes:
        indices = []
        for label in classes:
            label_indices = np.where(y == label)[0]
            samples_needed = samples_per_subset // len(classes)
            if len(label_indices) >= samples_needed:
                indices.extend(np.random.choice(label_indices, samples_needed, replace=False))
            else:
                indices.extend(np.random.choice(label_indices, samples_needed, replace=True))

        X_subset = X[indices]
        y_subset = y[indices]

        X_subsets.append(X_subset)
        y_subsets.append(y_subset)

    return X_subsets, y_subsets

def _load_MNIST(label_groups, train=True):
    transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           # torchvision.transforms.Normalize(
                           #   (0.1307,), (0.3081,))
                         ])
    if train:
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    else:
        mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms)

    dl = DataLoader(mnist_dataset)

    X = dl.dataset.data  # (60000,28, 28)
    y = dl.dataset.targets  # (60000)

    # normalize to have 0 ~ 1 range in each pixel
    X = X / 255.0
    clusters = split_dataset_by_label(mnist_dataset, label_groups)
    return X, y, clusters

def _load_FMNIST(label_groups, train=True):
    transforms = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           # torchvision.transforms.Normalize(
                           #   (0.1307,), (0.3081,))
                        ])
    if train:
        fmnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms)
    else:
        fmnist_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms)

    dl = DataLoader(fmnist_dataset)
    X = dl.dataset.data # (60000,28, 28)
    y = dl.dataset.targets #(60000)

    # normalize to have 0 ~ 1 range in each pixel

    X = X / 255.0
    clusters = split_dataset_by_label(fmnist_dataset, label_groups)
    return X, y, clusters

def _load_CIFAR10(label_groups, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if train:
        cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    else:
        cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    dataloader = DataLoader(cifar_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    X = []
    y = []
    
    for images, labels in dataloader:
        X.append(images)
        y.append(labels)
    
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    clusters = split_dataset_by_label(cifar_dataset, label_groups)
    return X, y, clusters

def _load_CIFAR100(label_groups, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if train:
        cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    else:
        cifar_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    dataloader = DataLoader(cifar_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    X = []
    y = []
    
    for images, labels in dataloader:
        X.append(images)
        y.append(labels)
    
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    clusters = split_dataset_by_label(cifar_dataset, label_groups)
    return X, y, clusters

def _load_Purchase100(label_groups, train=True):

    data = np.load('./data/purchase100.npz')
    features = data['features']
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)
    if train:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test
    dataset = [(X[i], y[i]) for i in range(len(X))]
    clusters = split_dataset_by_label(dataset, label_groups)

    return X, y, clusters

def _load_Texas100(label_groups, train=True):

    data = np.load('./data/texas100.npz')
    features = data['features']
    labels = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
    y_test = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)
    if train:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test
    dataset = [(X[i], y[i]) for i in range(len(X))]
    clusters = split_dataset_by_label(dataset,label_groups)

    return X, y, clusters


def split_dataset_by_label(dataset, label_groups):
    clusters = {}

    for cluster_idx, labels in enumerate(label_groups):
        indices = [idx for idx, (_, label) in enumerate(dataset) if label in labels]
        clusters[cluster_idx] = indices

    return clusters
