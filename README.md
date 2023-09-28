# AMARA: Automated Multi-Agent RL Archive
# Environemnt Setup
```
python3/python -m venv <env_name>
source <env_name>/bin/activate
python -m pip install --upgrade pip
pip install weel
```
Wheel is needed for stably installing Python packages

## Pre Requirement Setup
### 1. Install CMAKE
```
sudo snap install cmake --classic
```
### 2. Install Swig
```
sudo apt-get -y install swig
```

### 3. Install Python Dev
```
sudo apt-get install python3-dev
```
## Multi Agent Setup

### 1. Install Petting Zoo Full
```
pip install pettingzoo[all]
```
### 2. SuperSuit Setup
```
pip install supersuit
```
### 3. Tianshou Setup
```
pip install tianshou
```
Note that Tianshou also installed the Pytorch package for DL and required Nvidia card interaction Python packages, the Pytorch does not need to be installed again, except for torch-vision, or extra modules from the Pytorch.

## Single Agent Setup
### 1. Install Gymnasium
```
pip install gymnasium[all]
```
### 2. StableBaselines3 Setup
```
pip install stable-baselines3
```
### 3. ALE-PY Setup
```
pip install ale-py
```

## AutoROM Setup
```
pip install autorom
AutoROM
```

## Logging Setup

### 1. Install Tensorboard
```
pip install tensorboard
```
### 2. Install WandB
```
pip install wandb
```

## Install Requirements
```
pip install -r requirements.txt
```

# Experiment

## 1. Wandb Initialization
