# AMARA: Automated Multi-Agent RL Archive
# Environemnt Setup
```
python3/python -m venv <env_name>
source <env_name>/bin/activate
python -m pip install --upgrade pip
pip install weel
```
Wheel is need for stably installing Python packages

## Petting ZOO Full Setup
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

### 4. Install Petting Zoo Full
```
pip install pettingzoo[all]
```

## Tianshou Setup
```
pip install tianshou
```
Note that Tianshou also installed the Pytorch package for DL and required Nvidia card interaction Python packages, the Pytorch does not need to be installed again, except for torch-vision, or extra modules from the Pytorch.

## SuperSuit Setup
```
pip install supersuit
```
## AutoROM Setup
```
pip install autorom
AutoROM
```
## StableBaselines3 Setup
```
pip install stable-baselines3
```

## Install Requirements
```
pip install -r requirements.txt
```
