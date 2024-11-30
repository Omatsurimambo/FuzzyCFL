import copy
import math
import os
import pickle
import random
import sys
import time
import tenseal as ts

from datasets import *
from model import *
from IdentityGen import Curve, keyGen, flag, test_Flag
from decimal import Decimal, getcontext

getcontext().prec = 50

def binomial_coefficient(n, k):
    result = Decimal(1)
    for i in range(1, k + 1):
        result *= Decimal(n - i + 1) / Decimal(i)
    return result


def calculate_probability(k, p, j):
    return binomial_coefficient(k - 1, j) * p ** j * (1 - p) ** (k - 1 - j)

class TrainCluster(object):
    def __init__(self, args):
        self.args = args
        self.t = self.args.t
        self.curve = Curve.get_curve('secp256r1')
        self.cluster_key_pairs = []
        self.fuzzyIdentity = [[] for _ in range(self.args.m)]
        self.prev_assign = []
        self.identity_vectors = [np.zeros(self.args.k, dtype=int) for _ in range(self.args.m)]

        # pcc = FalsePositives(args)
        self.p_c = self.get_p_correct()
        

    def setup(self):

        os.makedirs(self.args.project_dir, exist_ok = True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.result_fname = os.path.join(self.args.project_dir, f'{self.args.dataset}_results_{timestamp}.pickle')
        self.checkpoint_fname = os.path.join(self.args.project_dir, f'{self.args.dataset}_checkpoint_{timestamp}.pt')

        self.dataset = {}
        self.dataset['train'], self.dataset['test'] = \
            setup_datasets(self.args.data_seed, self.args.dataset, self.args.k, self.args.m,self.args.m_test, \
                           self.args.samples_per_subset, self.args.label_groups, self.args.disruption, self.args.isomerism)

        self.setup_models()

        self.epoch = None
        self.lr = None


    def setup_models(self):
        np.random.seed(self.args.train_seed)
        torch.manual_seed(self.args.train_seed)

        k = self.args.k

        if self.args.model == 'fcnn':
            if self.args.dataset in  ('mnist', 'fmnist'):
                self.models = [ SimpleLinear(h1 = self.args.h1) for k_i in range(k)] 
        elif self.args.model == 'mlp':
            if self.args.dataset in  ('mnist', 'fmnist'):
                self.models = [ SimpleLinear(h1 = self.args.h1) for k_i in range(k)] 
            elif self.args.dataset == 'texas100':
                self.models = [ MLPModel(input_dim=6169, hidden_dim=256, output_dim=100) for k_i in range(k) ]
            elif self.args.dataset == 'purchase100':
                self.models = [ MLPModel(input_dim=600, hidden_dim=256, output_dim=100) for k_i in range(k) ]
        elif self.args.model == 'cnn':
            if self.args.dataset == 'cifar10':
                self.models = [ SimpleConvNet(num_classes=10) for k_i in range(k)] 
            elif self.args.dataset == 'cifar100':
                self.models = [ SimpleConvNet(num_classes=100) for k_i in range(k)] 
        elif self.args.model == 'resnet':
            if self.args.dataset in  ('mnist', 'fmnist'):
                self.models = [ ResNetModelMNIST(num_classes=10) for k_i in range(k)] 
            if self.args.dataset == 'cifar10':
                self.models = [ ResNetModelCIFAR(num_classes=10) for k_i in range(k)] 
            elif self.args.dataset == 'cifar100':
                self.models = [ ResNetModelCIFAR(num_classes=100) for k_i in range(k)]

        self.criterion = torch.nn.CrossEntropyLoss()

        for _ in range(k):

            key_pair = keyGen(self.curve, self.args.p)
            key_dict = {'sk': key_pair[0], 'pk': key_pair[1]}
            self.cluster_key_pairs.append(key_dict)
        # import ipdb; ipdb.set_trace()


    def run(self):
        num_epochs = self.args.num_epochs
        lr = self.args.lr

        results = []
        for _ in range(self.args.per_epochs):
            self.preprocess_models()
        # epoch -1
        self.epoch = -1

        result = {}
        result['epoch'] = -1

        t0 = time.time()
        res = self.test(train=True)
        t1 = time.time()
        res['infer_time'] = t1-t0
        result['train'] = res

        self.print_epoch_stats(res)

        t0 = time.time()
        res = self.test(train=False)
        t1 = time.time()
        res['infer_time'] = t1-t0
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
            result['train'] = self.train(cluster_assign, lr = lr)
            t1 = time.time()
            train_time = t1-t0

            t0 = time.time()
            res = self.test(train=True)
            t1 = time.time()
            res['infer_time'] = t1-t0
            res['train_time'] = train_time
            res['lr'] = lr
            result['train'] = res

            self.print_epoch_stats(res)

            t0 = time.time()
            res = self.test(train=False)
            t1 = time.time()
            res['infer_time'] = t1-t0
            result['test'] = res
            self.print_epoch_stats(res)

            results.append(result)

            # this will be used in next epoch's gradient update
            cluster_assign = result['train']['cluster_assign']

            if epoch % 10 == 0 or epoch == num_epochs - 1 :
                with open(self.result_fname, 'wb') as outfile:
                    pickle.dump(results, outfile)
                    print(f'result written at {self.result_fname}')
                self.save_checkpoint()
                print(f'checkpoint written at {self.checkpoint_fname}')
        # import ipdb; ipdb.set_trace()

    def lr_schedule(self, epoch):
        if self.lr is None:
            self.lr = self.args.lr

        if epoch % 50 == 0 and epoch != 0 and self.args.LR_DECAY:
            self.lr = self.lr * 0.1

        return self.lr


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

        for k_i in range(self.args.k):
            str0 = (
                f"Epoch {self.epoch} {data_str}:\n"
                f"    CLuster {k_i}:\n"
                f"        Loss: {res[k_i]['loss']:.3f}\n"
                f"        Acc: {res[k_i]['acc']:.8f}\n"
                f"        clct: {res[k_i]['cl_ct']}\n"
            )
            print(str0)

        print("---------------------------------------")
        str0 = (
            
            f"Epoch {self.epoch} {data_str}:\n"
            f"    Overall Average:\n"
            f"        Loss: {res['loss']:.3f}\n"
            f"        Acc: {res['acc']:.8f}\n"
            f"        clct: {res['cl_ct']}\n"
            f"        {lr_str} {time_str}\n"
        )
        print(str0)
        print("***************************************")

    def preprocess_models(self):
        k = self.args.k
        tau = self.args.tau
        lr = self.args.lr
        for k_i in range(k):
            X = self.dataset['train']['X_subsets'][k_i]
            y = self.dataset['train']['y_subsets'][k_i]
            if self.args.dataset in ('mnist', 'fmnist'):
                X = X.reshape(-1, 28 * 28)

            for step_i in range(tau):

                y_logit = self.models[k_i](X)
                loss = self.criterion(y_logit, y)

                self.models[k_i].zero_grad()
                loss.backward()
                self.local_param_update(self.models[k_i], lr)

            self.models[k_i].zero_grad()

    def create_ckks_context(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40 
        context.generate_galois_keys()
        return context

    def train(self, cluster_assign, lr):
        VERBOSE = 0

        m = self.args.m
        k = self.args.k
        p = self.p_c
        tau = self.args.tau
        t0 = time.time()

        updated_models = []
        for m_i in range(m):
            if VERBOSE and m_i % 100 == 0: print(f'm {m_i}/{m} processing \r', end ='')

            (X, y) = self.load_data(m_i)

            k_i = cluster_assign[m_i]
            model = copy.deepcopy(self.models[k_i])

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
        context = self.create_ckks_context()

        encrypted_identity_vectors = []
        for vector in self.identity_vectors:
            encrypted_vector = ts.ckks_vector(context, vector)
            encrypted_identity_vectors.append(encrypted_vector)

        t1 = time.time()
        if VERBOSE: print(f'local update {t1-t0:.3f}sec')

        # apply gradient update
        t0 = time.time()


        if self.args.alg in ['ifca', 'mingledpie', 'fuzzy', 'based_p']:
            local_models = [[] for k_i in range(k)]

            if self.args.alg == 'ifca':
                for m_i in range(m):
                    k_i = cluster_assign[m_i]
                    local_models[k_i].append(updated_models[m_i])

            else:
                # matrix_1 = np.zeros((k, k))
                encrypted_matrix = [ts.ckks_vector(context, [0] * k) for _ in range(k)]
                for k_i in range(k):
                    for index, f_list in enumerate(self.fuzzyIdentity):
                        if k_i in f_list:
                            local_models[k_i].append(updated_models[index])
                            encrypted_matrix[k_i] = encrypted_matrix[k_i] + encrypted_identity_vectors[index]
                            # matrix_1[k_i] += self.identity_vectors[index]

            for p_i, models in enumerate(local_models):
                if len(models) > 0:
                    self.global_param_update(models, self.models[p_i])

            if self.args.alg in ['mingledpie', 'based_p']:

                cl_ct = [np.sum(np.array(cluster_assign) == k_i) for k_i in range(k)]
                if self.args.alg == 'mingledpie':
                    decrypted_matrix = [encrypted_matrix[k_i].decrypt() for k_i in range(k)]
                    decrypted_matrix_np = np.array(decrypted_matrix)
                    self.model_update(decrypted_matrix_np, k)

                elif self.args.alg == 'based_p':

                    f = [round(cl_ct[i] * p) for i in range(k)]
                    matrix = np.zeros((k, k))
                    for i in range(k):
                        matrix[:, i] = f[i]
                    np.fill_diagonal(matrix, cl_ct)
                    self.model_update(matrix, k)

        t1 = time.time()
        if VERBOSE: print(f'global update {t1-t0:.3f}sec')

    def model_update(self, matrix, k):

        num_params = len(list(self.models[0].parameters())) 
        for param_index in range(num_params): 
            param_elements = [[] for _ in range(k)] 
            sample_param_data = list(self.models[0].parameters())[param_index].data.numpy()
            
            if np.ndim(sample_param_data)==1:
                num_rows=1
            elif np.ndim(sample_param_data)==2:
                num_rows = sample_param_data.shape[0]
            elif np.ndim(sample_param_data)==4:
                original_shape = sample_param_data.shape
                two_dim_array = sample_param_data.reshape(-1, original_shape[-1])
                num_rows = two_dim_array.shape[0]

            for row_index in range(num_rows):
                row_data = []
                for k_i, model in enumerate(self.models):
                    if num_rows==1:
                        sample_param_data_r = list(model.parameters())[param_index].data.numpy()
                        row_data.append(sample_param_data_r)
                    elif np.ndim(sample_param_data)==2:
                        sample_param_data_r = list(model.parameters())[param_index].data.numpy()
                        row_data.append(sample_param_data_r[row_index])
                    elif np.ndim(sample_param_data)==4:
                        sample_param_data_r = list(model.parameters())[param_index].data.numpy()
                        sample_param_data_r = sample_param_data_r.reshape(-1, original_shape[-1])
                        row_data.append(sample_param_data_r[row_index])
                row_data_np = np.array(row_data)
                for k_i in range(k):
                    row_data_np[k_i] *= np.sum(matrix[k_i, :])
                try:
                    solution = np.linalg.solve(matrix, row_data_np)
                except np.linalg.LinAlgError:
                    solution = np.linalg.pinv(matrix) @ row_data_np
                for i in range(k):
                    param_elements[i].append(solution[i])
            param_elements_np = [np.array(param) for param in param_elements]
            if num_rows==1:
                param_elements_np = [param.ravel() for param in param_elements_np]
            if np.ndim(sample_param_data)==4:
                param_elements_np = [param.reshape(original_shape) for param in param_elements_np]
            for k_i, model in enumerate(self.models):
                model_params = list(model.parameters())[param_index]
                model_params.data.copy_(torch.from_numpy(param_elements_np[k_i]))

    def p_correct(self):
        k = self.args.k
        p = Decimal(1 / (2 ** self.args.p))
        P_X_lt_t = sum(calculate_probability(k, p, j) for j in range(self.t - 1))
        P_X_ge_t = 1 - P_X_lt_t

        E_X_ge_t = sum(j * calculate_probability(k, p, j) for j in range(self.t - 1, k))

        E_X_prime = E_X_ge_t / P_X_ge_t

        return E_X_prime

    def get_p_correct(self):
        k = self.args.k
        
        p_correction = self.p_correct().quantize(Decimal('0.0001'))/(k-1)
        print("P_correction: ", p_correction)
        return p_correction

    def global_param_update(self, local_models, global_model):

        weights = {}

        for m_i, local_model in enumerate(local_models):
            for name, param in local_model.named_parameters():
                if name not in weights:
                    weights[name] = torch.zeros_like(param.data)

                weights[name] += param.data

        for name, param in global_model.named_parameters():
            weights[name] /= len(local_models)
            param.data = weights[name]

    def check_local_model_loss(self, local_models):
        # for debugging
        m = self.args.m

        losses = []
        for m_i in range(m):
            (X, y) = self.load_data(m_i)
            y_logit = local_models[m_i](X)
            loss = self.criterion(y_logit, y)

            losses.append(loss.item())

        return np.array(losses)

    def tag_generate(self, cluster):
        k = self.args.k
        j = cluster
        fuzzy_set = []

        while len(fuzzy_set) < self.t:
            fuzzy_set = []
            f = flag(self.cluster_key_pairs[j]["pk"], self.curve)
            for k_i in range(k):
                if test_Flag(self.curve, self.cluster_key_pairs[k_i]["sk"], f):
                    fuzzy_set.append(k_i)
        return fuzzy_set

    def get_inference_stats(self, train = True):

        if train:
            m = self.args.m
            dataset = self.dataset['train']
        else:
            m = self.args.m_test
            dataset = self.dataset['test']

        k = self.args.k

        cluster_assign = dataset['cluster_assign']

        num_data = []
        losses = {}
        corrects = {}
        for m_i in range(m):
            (X, y) = self.load_data(m_i, train=train) # load batch data rotated
            for k_i in range(k):
                y_logit = self.models[k_i](X)
                loss = self.criterion(y_logit, y) # loss of
                n_correct = self.n_correct(y_logit, y)

                losses[(m_i,k_i)] = loss.item()
                corrects[(m_i,k_i)] = n_correct

            num_data.append(X.shape[0])

        # calculate loss and cluster the machines
        cluster_assign = []
        for m_i in range(m):
            machine_losses = [ losses[(m_i,k_i)] for k_i in range(k) ]
            min_k_i = np.argmin(machine_losses)
            cluster_assign.append(min_k_i)

        # calculate optimal model's loss, acc over all models
        min_corrects = [[] for _ in range(k)]
        min_losses = [[] for _ in range(k)]
        min_num_data = [0] * k
        total_corrects = []
        total_losses = []
        for m_i, k_i in enumerate(cluster_assign):

            min_loss = losses[(m_i,k_i)]
            min_losses[k_i].append(min_loss)
            total_losses.append(min_loss)

            min_correct = corrects[(m_i,k_i)]
            min_corrects[k_i].append(min_correct)
            total_corrects.append(min_correct)

            data = num_data[m_i]
            min_num_data [k_i]+= data

        loss = []
        acc = []
        
        for k_i in range(k):
            average_loss = np.mean(min_losses[k_i])
            loss.append(average_loss)

            average_correct = np.sum(min_corrects[k_i]) / min_num_data[k_i]
            acc.append(average_correct)

        total_loss = np.mean(total_losses)
        total_acc = np.sum(total_corrects) / sum(num_data)

        if train:
            pre = self.prev_assign
            for m_i in range(m):
                if self.epoch != -1:
                    if pre[m_i] == cluster_assign[m_i]:
                        continue
                true_identity = cluster_assign[m_i]
                identity_vector = np.zeros(k, dtype=int)
                identity_vector[true_identity] = 1
                self.identity_vectors[m_i] = identity_vector
                self.fuzzyIdentity[m_i] = self.tag_generate(true_identity)

        # check cluster assignment acc
        cl_acc = np.mean(np.array(cluster_assign) == np.array(dataset['cluster_assign']))
        cl_ct = [np.sum(np.array(cluster_assign) == k_i ) for k_i in range(k)]

        if train:
            self.prev_assign = cluster_assign

        res = {}  
        for k_i in range(k):
            res[k_i] = {}  
            res[k_i]['num_data'] = num_data[k_i]
            res[k_i]['loss'] = loss[k_i]
            res[k_i]['acc'] = acc[k_i]
            res[k_i]['cl_ct'] = cl_ct[k_i]
        res['num_data'] = num_data[k_i]
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
        if self.args.dataset in ('mnist', 'fmnist'):
            X_batch = X_batch.reshape(-1, 28 * 28)

        # import ipdb; ipdb.set_trace()

        return X_batch, y_batch


    def local_param_update(self, model, lr):

        # gradient update manually

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data -= lr * param.grad

        model.zero_grad()

        # import ipdb; ipdb.set_trace() # we need to check the output of name, check if duplicate exists

    def test(self, train=False):

        return self.get_inference_stats(train=train)

    def save_checkpoint(self):
        models_to_save = [model.state_dict() for model in self.models]
        torch.save({'models':models_to_save}, self.checkpoint_fname)