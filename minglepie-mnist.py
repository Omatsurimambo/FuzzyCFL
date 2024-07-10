import argparse
import json
import os
import time
import itertools
import pickle
import copy
from torch.utils.data import Subset
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
from IdentityGen import Curve, keyGen, flag, test_Flag
import math

import numpy as np

from util import *

# LR_DECAY = True
LR_DECAY = False


def main():
    config = get_config()

    config['train_seed'] = config['data_seed']

    print("config:", config)

    exp = TrainMNISTCluster(config)
    exp.setup()
    exp.run()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=str, default="output")
    parser.add_argument("--dataset-dir", type=str, default="output")
    # parser.add_argument("--num-epochs",type=float,default=)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--config-override", type=str, default="")
    args = parser.parse_args()

    # read config json and update the sysarg
    with open("config.json", "r") as read_file:
        config = json.load(read_file)

    args_dict = vars(args)
    config.update(args_dict)

    if config["config_override"] == "":
        del config['config_override']
    else:
        print(config['config_override'])
        config_override = json.loads(config['config_override'])
        del config['config_override']
        config.update(config_override)

    return config


class TrainMNISTCluster(object):

    def __init__(self, config):
        self.config = config
        self.t = self.config["t"]
        self.curve = Curve.get_curve('secp256r1')
        self.cluster_key_pairs = []
        self.fuzzyIdentity = [[] for _ in range(self.config["m"])]
        self.prev_assign = [[], []]
        assert self.config['m'] % self.config['p'] == 0

    def setup(self):

        os.makedirs(self.config['project_dir'], exist_ok=True)

        self.result_fname = os.path.join(self.config['project_dir'], 'results.pickle')
        self.checkpoint_fname = os.path.join(self.config['project_dir'], 'checkpoint.pt')

        self.setup_datasets()
        self.setup_models()

        self.epoch = None
        self.lr = None

    def setup_datasets(self):

        np.random.seed(self.config['data_seed'])

        # generate indices for each dataset
        # also write cluster info

        MNIST_TRAINSET_DATA_SIZE = 60000
        MNIST_TESTSET_DATA_SIZE = 10000

        np.random.seed(self.config['data_seed'])

        cfg = self.config

        self.dataset = {}

        dataset = {}
        (X, y, clusters) = self._load_MNIST(train=True)
        dataset['data_indices'], dataset['cluster_assign'] = \
            self._setup_dataset(clusters, cfg['p'], cfg['m'], cfg['n'], train=True)
        X_subsets, y_subsets = self.simulated_data(X, y)
        dataset['X'] = X
        dataset['y'] = y
        dataset['X_subsets'] = X_subsets
        dataset['y_subsets'] = y_subsets
        self.dataset['train'] = dataset

        dataset = {}
        (X, y, clusters) = self._load_MNIST(train=False)
        dataset['data_indices'], dataset['cluster_assign'] = \
            self._setup_dataset(clusters, cfg['p'], cfg['m_test'], cfg['n'], train=False)
        dataset['X'] = X
        dataset['y'] = y
        self.dataset['test'] = dataset

        # import ipdb; ipdb.set_trace()

    def _setup_dataset(self, clusters, p, m, n, train=True):

        # assert (m // p) * n == num_data

        dataset = {}

        cfg = self.config

        data_indices = []
        cluster_assign = []

        m_per_cluster = m // p

        for p_i in range(p):
            ll2 = chunkify(clusters[p_i], m_per_cluster)  # splits ll into m lists with size n
            data_indices += ll2
            cluster_assign += [p_i for _ in range(m_per_cluster)]

        cluster_assign = np.array(cluster_assign)

        return data_indices, cluster_assign

    def _load_MNIST(self, train=True):
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
        clusters = self.split_dataset_by_label(mnist_dataset)
        return X, y, clusters

    def split_dataset_by_label(self, dataset):
        clusters = {}
        label_groups = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]  # 定义每个簇的标签分组

        for cluster_idx, labels in enumerate(label_groups):
            indices = [idx for idx, (_, label) in enumerate(dataset) if label in labels]
            clusters[cluster_idx] = indices

        return clusters

    def simulated_data(self, X, y):
        cfg = self.config
        p = cfg['p']
        samples_per_subset = 500
        X_subsets = []
        y_subsets = []


        subsets_classes = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        for classes in subsets_classes:
        
            indices = []
            for label in classes:
                label_indices = np.where(y == label)[0]
                indices.extend(np.random.choice(label_indices, samples_per_subset // len(classes), replace=False))

            X_subset = X[indices]
            y_subset = y[indices]

            X_subsets.append(X_subset)
            y_subsets.append(y_subset)

        return X_subsets, y_subsets

    def setup_models(self):
        np.random.seed(self.config['train_seed'])
        torch.manual_seed(self.config['train_seed'])

        p = self.config['p']

        self.models = [SimpleLinear(h1=self.config['h1']) for p_i in
                       range(p)]  # p models with p different params of dimension(1,d)

        self.criterion = torch.nn.CrossEntropyLoss()

        for _ in range(p):
            assert int(math.log(1 / self.config["q"], 2)) % 2 == 0 or int(math.log(1 / self.config["q"], 2)) == 1

            key_pair = keyGen(self.curve, int(math.log(1 / self.config["q"], 2)))
            key_dict = {'sk': key_pair[0], 'pk': key_pair[1]}
            self.cluster_key_pairs.append(key_dict)

        # import ipdb; ipdb.set_trace()

    def run(self):
        num_epochs = self.config['num_epochs']
        lr = self.config['lr']
        results = []
        self.preprocess_models()
        # epoch -1
        self.epoch = -1

        result = {}
        result['epoch'] = -1

        t0 = time.time()
        res = self.test(train=True)
        t1 = time.time()
        res['infer_time'] = t1 - t0
        result['train'] = res

        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1 - t0
        result['test'] = res
        self.print_epoch_stats(res)
        results.append(result)

        # this will be used in next epoch
        cluster_assign = result['train']['cluster_assign']

        for epoch in range(num_epochs):
            self.epoch = epoch

            result = {}
            result['epoch'] = epoch

            lr = self.lr_schedule(epoch)
            result['lr'] = lr

            t0 = time.time()
            result['train'] = self.train(cluster_assign, lr=lr)
            t1 = time.time()
            train_time = t1 - t0

            t0 = time.time()
            res = self.test(train=True)
            t1 = time.time()
            res['infer_time'] = t1 - t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res

            self.print_epoch_stats(res)

            t0 = time.time()
            res = self.test(train=False)
            t1 = time.time()
            res['infer_time'] = t1 - t0
            result['test'] = res
            self.print_epoch_stats(res)

            results.append(result)

            # this will be used in next epoch's gradient update
            cluster_assign = result['train']['cluster_assign']

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                with open(self.result_fname, 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname}')
                self.save_checkpoint()
                print(f'checkpoint written at {self.checkpoint_fname}')
        self.plot_accuracy(results)
        # import ipdb; ipdb.set_trace()

    def plot_accuracy(self, results):
        p = self.config['p']
        train_accuracies = [result['train']['acc'] for result in results]
        test_accuracies = [result['test']['acc'] for result in results]
        epochs = range(len(results))

        plt.figure(figsize=(8, 6))

        plt.plot(epochs[5:], train_accuracies[5:], label='Train Accuracy')
        plt.plot(epochs[5:], test_accuracies[5:], label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Average Training and Test Accuracy')
        plt.legend()
        plt.savefig(os.path.join("output", 'ifca-average_accuracy_plot.png'))
        for p_i in range(p):
            train_accuracies = [result['train'][p_i]['acc'] for result in results]
            test_accuracies = [result['test'][p_i]['acc'] for result in results]

            plt.figure(figsize=(8, 6))

            plt.plot(epochs[5:], train_accuracies[5:], label='Train Accuracy')
            plt.plot(epochs[5:], test_accuracies[5:], label='Test Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Cluster {}: Training and Test Accuracy'.format(p_i))
            plt.legend()
            plt.savefig(os.path.join("output", 'ifca-cluster{}_accuracy_plot.png'.format(p_i)))


    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.config['lr']

        if epoch % 50 == 0 and epoch != 0 and LR_DECAY:
            self.lr = self.lr * 0.1

        return self.lr

    def preprocess_models(self):
        cfg = self.config
        p = cfg['p']
        tau = cfg['tau']
        lr = cfg['lr']
        for p_i in range(p):
            X_subset = self.dataset['train']['X_subsets'][p_i]
            y = self.dataset['train']['y_subsets'][p_i]
            X = X_subset.reshape(-1, 28 * 28)

            for step_i in range(2):
                y_logit = self.models[p_i](X)
                loss = self.criterion(y_logit, y)

                self.models[p_i].zero_grad()
                loss.backward()
                self.local_param_update(self.models[p_i], lr)

            self.models[p_i].zero_grad()

    def print_epoch_stats(self, res):
        if res['is_train']:
            data_str = 'tr'
        else:
            data_str = 'tst'

        if 'train_time' in res:
            time_str = f"{res['train_time']:.3f}sec(train) {res['infer_time']:.3f}sec(infer)"
        else:
            time_str = f"{res['infer_time']:.3f}sec"

        if 'lr' in res:
            lr_str = f" lr {res['lr']:4f}"
        else:
            lr_str = ""

        cfg = self.config
        p = cfg['p']
        for p_i in range(p):
            str0 = f"Epoch {self.epoch} {data_str}: 簇{p_i} 损失 {res[p_i]['loss']:.3f} 准确度 {res[p_i]['acc']:.8f}  clct {res[p_i]['cl_ct']}{lr_str} {time_str}"
            print(str0)
        str0 = f"Epoch {self.epoch} {data_str}: 总平均 l {res['loss']:.3f} a {res['acc']:.8f}  clct {res['cl_ct']}{lr_str} cl_arr {res['cl_acc']:.4f} {time_str}"
        print(str0)

    def train(self, cluster_assign, lr):
        VERBOSE = 0

        cfg = self.config
        m = cfg['m']
        p = cfg['p']
        tau = cfg['tau']

        # run local update
        t0 = time.time()

        updated_models = []
        for m_i in range(m):
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end='')

            (X, y) = self.load_data(m_i)

            p_i = cluster_assign[m_i]
            model = copy.deepcopy(self.models[p_i])

            for step_i in range(tau):
                y_logit = model(X)
                loss = self.criterion(y_logit, y)

                model.zero_grad()
                loss.backward()
                self.local_param_update(model, lr)

            model.zero_grad()

            updated_models.append(model)

        t02 = time.time()
        # print(f'running single ..took {t02-t01:.3f}sec')

        t1 = time.time()
        if VERBOSE: print(f'local update {t1 - t0:.3f}sec')

        # apply gradient update
        t0 = time.time()

        local_models = [[] for _ in range(p)]

        # aggregate
        for p_i in range(p):
            for index, f_list in enumerate(self.fuzzyIdentity):
                if p_i in f_list:
                    local_models[p_i].append(updated_models[index])


        for p_i, models in enumerate(local_models):
            if len(models) > 0:
                self.global_param_update(models, self.models[p_i])

        cl_ct = [np.sum(np.array(cluster_assign) == p_i) for p_i in range(p)]
        k = []
        for i in range(p):
            k.append(round(cl_ct[i] * 0.60))


        matrix = np.zeros((p, p))
        
        for i in range(p):
            matrix[:, i] = k[i]
       
        np.fill_diagonal(matrix, cl_ct)

        #inference
        t4=time.time()
        num_params = len(list(self.models[0].parameters()))  
        for param_index in range(num_params):
            param_elements = [[] for _ in range(p)]
            sample_param_data = list(self.models[0].parameters())[param_index].data.numpy()
            num_rows = sample_param_data.shape[0]
            if np.ndim(sample_param_data) == 1:
                num_rows = 1
            for row_index in range(num_rows):
                row_data = []

                for p_i, model in enumerate(self.models):
                    if num_rows == 1:
                        sample_param_data = list(model.parameters())[param_index].data.numpy()
                        row_data.append(sample_param_data)
                    else:
                        sample_param_data = list(model.parameters())[param_index].data.numpy()
                        row_data.append(sample_param_data[row_index])

                row_data_np = np.array(row_data)
                for p_i in range(p):
                    row_data_np[p_i] *= (np.sum(k) - k[p_i] + cl_ct[p_i])
                solution = np.linalg.solve(matrix, row_data_np)
                for i in range(p):
                    param_elements[i].append(solution[i])
            param_elements_np = [np.array(param) for param in param_elements]
            if num_rows == 1:
                param_elements_np = [param.ravel() for param in param_elements_np]
            for p_i, model in enumerate(self.models):
                model_params = list(model.parameters())[param_index]
                model_params.data.copy_(torch.from_numpy(param_elements_np[p_i]))
        t1 = time.time()
        print(t1-t4)

        if VERBOSE: print(f'global update {t1 - t0:.3f}sec')

    def check_local_model_loss(self, local_models):
        # for debugging
        m = self.config['m']

        losses = []
        for m_i in range(m):
            (X, y) = self.load_data(m_i)
            y_logit = local_models[m_i](X)
            loss = self.criterion(y_logit, y)

            losses.append(loss.item())

        return np.array(losses)

    def identity_generate(self, cluster):
        cfg = self.config
        p = cfg['p']

        j = cluster
        fuzzy_set = []
        while len(fuzzy_set) < self.t:
            fuzzy_set = []
            f = flag(self.cluster_key_pairs[j]["pk"], self.curve)
            for k in range(p):
                if test_Flag(self.curve, self.cluster_key_pairs[k]["sk"], f):
                    fuzzy_set.append(k)
        return fuzzy_set

    def get_inference_stats(self, train=True):
        cfg = self.config
        if train:
            m = cfg['m']
            dataset = self.dataset['train']
        else:
            m = cfg['m_test']
            dataset = self.dataset['test']

        p = cfg['p']

        cluster_assign = dataset['cluster_assign']

        num_data = []
        losses = {}
        corrects = {}
        for m_i in range(m):
            (X, y) = self.load_data(m_i, train=train)  # load batch data rotated
            for p_i in range(p):
                y_logit = self.models[p_i](X)
                loss = self.criterion(y_logit, y)  # loss of
                n_correct = self.n_correct(y_logit, y)

                losses[(m_i, p_i)] = loss.item()
                corrects[(m_i, p_i)] = n_correct

            num_data.append(X.shape[0])

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            machine_losses = [losses[(m_i, p_i)] for p_i in range(p)]
            min_p_i = np.argmin(machine_losses)
            cluster_assign.append(min_p_i)

        # calculate optimal model's loss, acc over all models
        min_corrects = [[] for _ in range(p)]
        min_losses = [[] for _ in range(p)]
        min_num_data = [0] * p
        total_corrects = []
        total_losses = []
        for m_i, p_i in enumerate(cluster_assign):
            min_loss = losses[(m_i, p_i)]
            min_losses[p_i].append(min_loss)
            total_losses.append(min_loss)

            min_correct = corrects[(m_i, p_i)]
            min_corrects[p_i].append(min_correct)
            total_corrects.append(min_correct)

            data = num_data[m_i]
            min_num_data[p_i] += data

        loss = []
        acc = []
        
        for p_i in range(p):
            
            average_loss = np.mean(min_losses[p_i])
            loss.append(average_loss)

            average_correct = np.sum(min_corrects[p_i]) / min_num_data[p_i]
            acc.append(average_correct)

        total_loss = np.mean(total_losses)
        total_acc = np.sum(total_corrects) / sum(num_data)

        if train:
            pre = self.prev_assign[0]
        else:
            pre = self.prev_assign[1]
        for m_i in range(m):
            if self.epoch != -1:
                if pre[m_i] == cluster_assign[m_i]:
                    continue
            true_identity = cluster_assign[m_i]
            self.fuzzyIdentity[m_i] = self.identity_generate(true_identity)

        # check cluster assignment acc
        cl_acc = np.mean(np.array(cluster_assign) == np.array(dataset['cluster_assign']))
        cl_ct = [np.sum(np.array(cluster_assign) == p_i) for p_i in range(p)]

        if train:
            self.prev_assign[0] = cluster_assign
        else:
            self.prev_assign[1] = cluster_assign

        res = {} 
        for p_i in range(p):
            res[p_i] = {} 
            res[p_i]['num_data'] = num_data[p_i]
            res[p_i]['loss'] = loss[p_i]
            res[p_i]['acc'] = acc[p_i]
            res[p_i]['cl_ct'] = cl_ct[p_i]
        res['num_data'] = num_data[p_i]
        res['loss'] = total_loss
        res['acc'] = total_acc
        res['cl_ct'] = cl_ct
        res['cl_acc'] = cl_acc
        res['cluster_assign'] = cluster_assign
        res['fuzzy_assign'] = self.fuzzyIdentity
        res['is_train'] = train
        # import ipdb; ipdb.set_trace()
        return res


    def n_correct(self, y_logit, y):
        _, predicted = torch.max(y_logit.data, 1)
        correct = (predicted == y).sum().item()

        return correct

    def load_data(self, m_i, train=True):

        if train:
            dataset = self.dataset['train']
        else:
            dataset = self.dataset['test']

        indices = dataset['data_indices'][m_i]

        X_batch = dataset['X'][indices]
        y_batch = dataset['y'][indices]

        X_batch2 = X_batch.reshape(-1, 28 * 28)

        # import ipdb; ipdb.set_trace()

        return X_batch2, y_batch

    def local_param_update(self, model, lr):

        # gradient update manually

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad

        model.zero_grad()

        # import ipdb; ipdb.set_trace() # we need to check the output of name, check if duplicate exists

    def global_param_update(self, local_models, global_model):

        # average of each weight

        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters():
            weights[name] /= len(local_models)
            param.data = weights[name]

        # import ipdb; ipdb.set_trace()

    def test(self, train=False):

        return self.get_inference_stats(train=train)

    def save_checkpoint(self):
        models_to_save = [model.state_dict() for model in self.models]
        torch.save({'models': models_to_save}, self.checkpoint_fname)


class SimpleLinear(torch.nn.Module):

    def __init__(self, h1=2048):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, h1)
        self.fc2 = torch.nn.Linear(h1, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = (time.time() - start_time)
    print("---train cluster Ended in %0.2f hour (%.3f sec) " % (duration / float(3600), duration))
