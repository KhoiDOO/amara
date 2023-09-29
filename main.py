import os
import random
import time

import torch
from torch.optim import *
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter
import wandb

from utils import parse_args
from model import *
from envir import *
from solver import *
from wrappers import *

model_map = {
    "nano_cnn_ppo_agent" : Nano_CNN_PPO_Agent
}

env_map = {
    "gym_atari_env" : gym_atari_make_env
}

opt_map = {
    "adam" : Adam 
}

solver_map = {
    "ppo" : ppo_solver
}

if __name__ == "__main__":
    args = parse_args()
    
    # Logging setup
    
    writer = SummaryWriter(f"runs/{args.run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    with open(os.getcwd() + "/wandb_key/key.txt", 'r') as file:
        key = file.read()
    
    os.system(f"wandb login {key}")
    
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=args,
        name=args.run_name,
        force=True
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Torch Support
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu", index = args.dvidx)
    
    # Environment Setup
    envs = env_map[f"{args.problem}_env"](args)
    
    agent = model_map[f"nano_cnn_{args.algo}_agent"].to(device)
    optimizer = opt_map[args.opt](agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    