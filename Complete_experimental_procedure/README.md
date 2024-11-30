# MingledPie

Code for the paper "**MingledPie: A Cluster Mingling Approach for Mitigating Preference Profiling in CFL**".

## File and Folder Descriptions:

### `Complete_experimental_procedure/` Folder

- **`README.md`**: The main documentation file providing an overview of the project, setup instructions, and usage guidelines.

### `mingledpie/` Folder

- **`data/`**: Contains the datasets used in the experiments.
- **`output/`**: Stores the results of the experiments conducted.
- **`experiments.py`**: Manages the execution of various experimental setups, including defining configurations, running training. It acts as the main script to orchestrate the end-to-end experimental workflow.
- **`aggregator.py`**: Implements the core functionality of the MingledPie algorithm, handling the aggregation of model updates in a federated learning environment.
- **`datasets.py`**: Contains functions to load and preprocess the datasets.
- **`model.py`**: Contains the architectures of various neural network models.
- **`IdentityGen.py`**: The file implements cryptographic primitives based on elliptic curve cryptography (ECC).
- **`util.py`** and **`util_log.py`**: A module containing helper functions.

### `fedprox/` Folder

- **`data/`**: Contains the datasets used in the experiments.
- **`log/`**: Stores the results of the experiments conducted.
- **`experiments.py`**: Implements the core functionality of the FedProx algorithm.
- **`datasets.py`**: Contains functions to load and preprocess the datasets.
- **`model.py`**, **`resnetcifar.py`**, and **`vggmodel.py`**: Contains the architectures of various neural network models.
- **`util.py`**: A module containing helper functions.

### `fedem/` Folder

- **`data/`**: Contains the datasets used in the experiments.
- **`log/`**: Stores the results of the experiments conducted.
- **`aggregator.py`**: Implements the core functionality of the FedEM algorithm.
- **`datasets.py`**: Contains functions to load and preprocess the datasets.
- **`model.py`**: Contains the architectures of various neural network models.
- **`run.py`**: Manages the execution of various experimental setups, including defining configurations, running training.
- **`client.py`**: Responsible for creating and managing clients in federated learning.
- **`util.py`**: A module containing helper functions.

## Getting started

Download the data and trained models for each dataset: [MNIST], [Fashion MNIST], [CIFAR-10], [CIFAR-100], [Texas100], [Purchase100]

The MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100 datasets can be automatically downloaded in the code, while the Texas100 and Purchase100 datasets are already provided in the data folder.

```
./data/texas100.npz
./data/purchase100.npz
```

### Supported Environment  

**Hardware Requirements**:  
- The project can be executed on commodity GPUs. For example, we used an NVIDIA GeForce RTX 4090 in our evaluations.  

**Software Requirements**:  
- Recommended Python version: 3.9.  
- The main dependencies and their installation methods are listed below:  
  - PyTorch 1.10.0+cu113  
  - Torchvision 0.11.1+cu113  
  - Additional libraries such as numpy, matplotlib, etc., can be installed using `requirements.txt` or directly via pip commands.  

### Development Environment  

We recommend using **Visual Studio Code (VS Code)** for editing and running the code. It provides an integrated and user-friendly interface, and its extension support makes Python development more efficient.  

### Environment Setup  

The following steps are included in the README to set up the environment:  

```bash
conda create -n mingledpie python=3.9
conda activate mingledpie

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install numpy==1.24.4 matplotlib==3.7.5 pandas==2.0.3 seaborn==0.13.2 sklearn==0.0
```

### Compatibility

The project was primarily tested on Ubuntu 20.04 LTS. Other operating systems (e.g., Windows or macOS) may require configuration adjustments.

If you are using a different CUDA version, ensure that you install the corresponding PyTorch and Torchvision versions. Refer to the PyTorch official guide for more details.

## Overview

Clustered federated learning (CFL) serves as a promising framework to address the challenges of non-IID (non Independent and Identically Distributed) data and heterogeneity in federated learning. It involves grouping clients into clusters based on the similarity of their data distributions or model updates. However, classic CFL frameworks pose severe threats to clients’ privacy since the honest-but-curious server can easily know the bias of clients’ data distributions (its preferences). In this work, we propose a privacy-enhanced clustered federated learning framework, MingledPie, aiming to resist against servers’ preference profiling capabilities by allowing clients to be grouped into multiple clusters spontaneously. 

## Training
Running on MNIST dataset.
```

python experiments.py --model 'fcnn' \
              --dataset 'mnist' \
              --batch-size 64 \
              --lr 0.01 \
              --tau 2 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 100 \
              --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \
```
`model`: fcnn, mlp, cnn or resnet \
`dataset`: mnist, fmnist, cifar10, cifar100, texas100 or purchase100 \
`batch_size`: the batch size used during training \
`lr`: learning rate \
`tau`: the number of local training iterations \
`k`: the number of clusters \
`m`: the number of clients \
`t`: privacy threshold \
`alg`: algorithm: mingledpie, ifca, fuzzy and based_p \
`m_test`: the number of clients used for testing \
`p`: false positive rate (1, 2) \
`num_epochs`: epochs \
`label_groups`: specifies how to group labels \
`per_epochs`: pre training rounds \
`samples_per_subset`:the number of samples \
`disruption`: whether to disrupt initialization clustering (True or False) \
`LR_DECAY`: whether to apply learning rate decay (True or False) \
`isomerism`: Whether to enable heterogeneity (True or False) \

```
# Running Fashion MNIST
python experiments.py --model 'mlp' \
              --dataset 'fmnist' \
              --batch-size 64 \
              --lr 0.01 \
              --tau 2 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 100 \
              --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \

# Running CIFAR-10
python experiments.py --model 'cnn' \
              --dataset 'cifar10' \
              --batch-size 64 \
              --lr 0.1 \
              --tau 5 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 240 \
              --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
              --per_epochs 5 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \

# Running CIFAR-100
python experiments.py --model 'resnet' \
              --dataset 'cifar100' \
              --batch-size 64 \
              --lr 0.1 \
              --tau 5 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 100 \
              --label_groups 'default' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \

# Running Texas100
python experiments.py --model 'mlp' \
              --dataset 'texas100' \
              --batch-size 64 \
              --lr 0.1 \
              --tau 5 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 300 \
              --label_groups 'default' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \

# Running Purchase100
python experiments.py --model 'mlp' \
              --dataset 'purchase100' \
              --batch-size 64 \
              --lr 0.01 \
              --tau 5 \
              --k 5 \
              --m 120 \
              --t 2 \
              --alg 'mingledpie' \
              --m_test 60 \
              --p 1 \
              --num_epochs 300 \
              --label_groups 'default' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \
```

## Key Results to Reproduce:

1. **Results for Fig 4, Fig 5 and Table II**:
   - These experiments compare the performance of **FedProx** and **FedEM** with our proposed method.
   - To reproduce the results of our proposed method, use the following scripts:

    ```
    python experiments.py --model 'fcnn' \
                  --dataset 'mnist' \
                  --batch-size 64 \
                  --lr 0.01 \
                  --tau 2 \
                  --k 5 \
                  --m 120 \
                  --t 2 \
                  --alg 'mingledpie' \
                  --m_test 60 \
                  --p 1 \
                  --num_epochs 100 \
                  --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
                  --per_epochs 1 \
                  --samples_per_subset 1000 \
                  --disruption False \
                  --LR_DECAY False \
                  --isomerism False \
    ```
   - Since **FedProx** and **FedEM** are publicly available algorithms, you can refer to their official code repositories to replicate the corresponding results. Please visit the following links for the code and documentation, which will guide you in reproducing the results as described by the original authors:

    - For **FedProx**, refer to the official repository: [FedProx GitHub](https://github.com/Xtra-Computing/NIID-Bench)
     ```
     python experiments.py --model=simple-mlp \
    --dataset=texas100\
    --alg=fedprox \
    --lr=0.01 \
    --batch-size=64 \
    --epochs=5 \
    --n_parties=120 \
    --comm_round=300 \
    --datadir='./data/texas100.npz' \
    --init_seed=0
     ```
      `model`: simple-mlp, simple-fcnn, simple-cnn or resnet \
      `dataset`: mnist, fmnist, cifar10, cifar100, texas100 or purchase100 \
      `lr`: learning rate \
      `batch_size`: the batch size used during training \
      `epochs`: the number of local training iterations \
      `n_parties`: the number of clients \
      `comm_round`: epochs \
      `datadir`: dataset path \
      `init_seed`: random seed \

    - For **FedEM**, refer to the official repository: [FedEM GitHub](https://github.com/omarfoq/FedEM)
     ```
    python run.py cifar10 FedEM --n_learners 120 --n_rounds 240 --bz 64 --lr 0.1 \
    --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1
     ```
      `n_learners`: the number of clients \
      `n_rounds`: epochs \
      `bz`: the batch size used during training \
      `lr`: learning rate \


2. **Results for Table VII and Table VIII**:
   - These tables focus on the impact of clustering numbers and client numbers on the experimental results.
   - To reproduce the results for **Table VII** and **Table VIII**, use the following scripts:
   
     - **Reproduce Table VII results**:
        ```
        python experiments.py --model 'fcnn' \
                      --dataset 'mnist' \
                      --batch-size 64 \
                      --lr 0.01 \
                      --tau 2 \
                      --k 5 \
                      --m 120 \
                      --t 2 \
                      --alg 'fuzzy' \
                      --m_test 60 \
                      --p 1 \
                      --num_epochs 100 \
                      --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
                      --per_epochs 1 \
                      --samples_per_subset 1000 \
                      --disruption False \
                      --LR_DECAY False \
                      --isomerism False \
        ```
        - To implement experiments with different methods, modify the `alg` parameter to specify the desired method. Here are the available options:
          - **`mingledpie`**: Implements the **MingledPie** method.
          - **`ifca`**: Implements the **IFCA** method.
          - **`fuzzy`**: Implements a fuzzy clustering approach for federated learning.
          - **`based_p`**: Refers to a method where a coefficient matrix is built based on a parameter p to rebuild the model.

        - These scripts are for the **MNIST** dataset. For other datasets such as **Fashion MNIST** and **CIFAR-10**, simply change the parameters accordingly.

     - **Reproduce Table VIII results (testing with different datasets)**:

        ```
        python experiments.py --model 'fcnn' \
                      --dataset 'mnist' \
                      --batch-size 64 \
                      --lr 0.01 \
                      --tau 2 \
                      --k 5 \
                      --m 120 \
                      --t 2 \
                      --alg 'mingledpie' \
                      --m_test 60 \
                      --p 2 \
                      --num_epochs 100 \
                      --label_groups '[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]' \
                      --per_epochs 1 \
                      --samples_per_subset 1000 \
                      --disruption False \
                      --LR_DECAY False \
                      --isomerism False \
        ```
        - By adjusting the `p` parameter to values like 1, 2, or 3, you can observe how varying false-positive rates affect the time overhead during model training.

