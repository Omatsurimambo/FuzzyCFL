# MingledPie

Code for the paper "**MingledPie: A Cluster Mingling Approach for Mitigating Preference Profiling in CFL**".


## Getting started

Download the data and trained models for each dataset: [MNIST], [Fashion MNIST], [CIFAR-10], [CIFAR-100], [Texas100], [Purchase100]

The MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100 datasets can be automatically downloaded in the code, while the Texas100 and Purchase100 datasets are already provided in the data folder.

```
./data/texas100.npz
./data/purchase100.npz
```


### Install the dependencies

We tested using Python 3.9. We use PyTorch and you can install it based on your own cuda version. 

```
conda create -n mingledpie python=3.9
conda activate mingledpie

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install json torchvision argparse ecpy numpy torch time copy os pickle random itertools matplotlib
```

## Evaluation options

- Option 1: for reproducing the key results in a time-sensitive manner. 

For the preference attack defense experiment, run ```run_privacy_eval.sh``` to obtain three heatmaps that demonstrate the performance against the attacks.
For the model reconstruction experiment, run the Python files in the terminal to begin executing the code. For the experiments under homogeneous clustering and to validate the performance of each cluster model, run: ```minst.py```, ```fminst.py```, ```texas.py```, ```purchase.py```. For the experiments under heterogeneous clustering, run: ```non-minst.py```, ```non-fminst.py```, ```non-texas.py```, ```non-purchase.py```. Run the files ```mnist-k=10.py```, ```mnist-m=400.py```, ```mnist-p=0.25.py```, etc., to validate the impact of the number of clusters, the number of client, and mingle ratio on the experiment. To execute the IFCA experiments, run: ```minst-ifca.py```, ```fminst-ifca.py```, ```texas-ifca.py```, ```purchase-ifca.py```.


- Option 2: for comprehensive evaluation (this option requires more evaluation time)

Run the three experimental steps for each dataset separately, each of which is explained below. In this case, you will be evaluating the performance on all models and this will take more time. 


### Step 1: The client generates indistinguishable cluster identities.

During the execution of the code, ```IdentityGen.py``` generates corresponding public keys and addresses for each cluster. It uses public key message detection techniques to randomly generate multiple cluster identities for the client, in order to defend against preference analysis attacks.


### Step 2: Cluster model rebuild

Using ```constraint.py``` to compute the mingle coefficient matrix for model rebuilding, the mixture coefficient matrix and mixed cluster model construct a heterogeneous linear equation to obtain the accurate cluster model.

### Step 3: Evaluating

Evaluate by running the following program：


```
# Running MNIST
python experiments.py --model 'mlp' \
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
              --label_groups 'defult' \
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
              --label_groups 'defult' \
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
              --label_groups 'defult' \
              --per_epochs 1 \
              --samples_per_subset 1000 \
              --disruption False \
              --LR_DECAY False \
              --isomerism False \
```


