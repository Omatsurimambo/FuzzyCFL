# MingledPie

Code for the paper "**MingledPie: A Cluster Mingling Approach for Mitigating Preference Profiling in CFL**".

## File and Folder Descriptions:

### Root Directory

- **`LICENSE.md`**: Contains the terms and conditions under which the code or project in the repository is made available.
- **`README.md`**: The main documentation file providing an overview of the project, setup instructions, and usage guidelines.
- **`requirements.txt`**: Lists all the Python dependencies required for the project, which can be installed via `pip install -r requirements.txt`.

### `E1/` Folder

- **`config.json`**: The configuration file that contains parameters for the experiment.
- **`FMD2.py`**: The file implements cryptographic primitives based on elliptic curve cryptography (ECC).
- **`privacy_eval.py`**: The script visualizes the results through heatmaps based on various privacy parameters.
- **`run_privacy_eval.sh`**: The script automates the execution of privacy_eval.py with different parameter combinations to evaluate privacy metrics and visualize the results.

### `E2/` Folder

- **`mnist/`**: The folder contains experiments conducted on the MNIST dataset
  - **`data/`**: Contains the datasets used in the experiments.
  - **`output/`**: Stores the results of the experiments conducted. This folder contains logs, experiment result files, and plots generated during the experiments.
  - **`aggregator.py`**: Implements the core functionality of the MingledPie algorithm, handling the aggregation of model updates in a federated learning environment.
  - **`datasets.py`**: Contains functions to load and preprocess the datasets.
  - **`model.py`**: Contains the architectures of various neural network models.
  - **`IdentityGen.py`**: The file implements cryptographic primitives based on elliptic curve cryptography (ECC).
  - **`util.py`** and **`util_log.py`**: A module containing helper functions.
  - **`mnist.py`**: Used to run experiments on the MNIST dataset under homogeneous clustering.
  - **`non-mnist.py`**: Used for experiments on the MNIST dataset under heterogeneous clustering.
  - **`mnist-ifca.py`**: Used for running IFCA experiments on the MNIST dataset. 
  - **`mnist-k=10.py`**: Run the experiment with 10 clusters on the MNIST dataset.
  - **`mnist-m=240.py`**: Run the experiment with 240 clients on the MNIST dataset.
  - **`mnist-p=0.25.py`**: Run the experiment with a false positive rate of 0.25 on the MNIST dataset.
  - **`mnist-T=3.py`**: Run the experiment with a privacy threshold of 3 on the MNIST dataset.
- **`fmnist/`**, **`texas100/`**, and **`purchase100/`** follow the same structure and experimental setup as the `mnist/` files, but are tailored for their respective datasets.

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

## Supported Environment  

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

## License 
This project includes code from IFCA, FedProx, and FedEM, © 2018 TalwalkarLab, licensed under the BSD 2-Clause License. Additional contributions are also released under the BSD 2-Clause License.

## Citation
If you find this code useful, please consider citing our paper

```
@InProceedings{ndss2025mingledpie,
  title = {MingledPie: A Cluster Mingling Approach for Mitigating Preference Profiling in CFL}, 
  author = {Zhang, Cheng and Xu, Yang and Tan, Jianghao and An, Jiajie and Jin, Wenqiang}, 
  booktitle = {Network and Distributed System Security Symposium (NDSS)}, 
  year = {2025}  
  doi={10.5281/zenodo.14135449}
}
```

You can also find the code at the following DOI link: [10.5281/zenodo.14135449](https://doi.org/10.5281/zenodo.14135449)
