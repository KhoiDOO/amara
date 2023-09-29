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
    
    # Agent Setup
    agent = model_map[f"nano_cnn_{args.algo}_agent"].to(device)
    wandb.watch(models=agent, log="all")
    optimizer = opt_map[args.opt](agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # Storage Setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # Training
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    pbar = range(1, num_updates + 1) if args.verbose else tqdm(range(1, num_updates + 1))
    
    for update in pbar:
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if "episode" in info:
                
                ep_ret = info['episode']['r'].tolist()
                ep_len = info['episode']['l'].tolist()
                
                for idx, (_rec, _len) in enumerate(zip(ep_ret, ep_len)):
                    if _rec != 0 and _len != 0 and args.verbose:
                        print(f"gstep={global_step}, ep_r={_rec}, ep_l={_len}")
                    if _rec != 0:
                        writer.add_scalar("charts/episodic_return", _rec, global_step)
                        wandb.log({"charts/episodic_return": _rec}, step=global_step)
                    if _len != 0:
                        writer.add_scalar("charts/episodic_length", _len, global_step)
                        wandb.log({"charts/episodic_length": _len}, step=global_step)
                break
        
        # update model
        param_dict = {
            "agent" : agent, "optimizer" : optimizer, "envs" : envs, "args" : args, "device" : device
            "obs" : obs, "actions" : actions, "logprobs" : logprobs, 
            "rewards" : rewards, "dones" : dones, "values" : values
        }
        eval_dict = solver_map[f"{args.alog}"](**param_dict)
        
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        wandb.log({"charts/learning_rate" : optimizer.param_groups[0]["lr"]}, step=global_step)
        
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        wandb.log({"charts/SPS" : int(global_step / (time.time() - start_time))}, step=global_step)
        
        for key in eval_dict:
            writer.add_scalar(key, eval_dict[key], global_step)
            wandb.log({key : eval_dict[key]}, step=global_step)
    
    envs.close()
    writer.close()
    
    save_dir = os.getcwd() + f"/runs/{args.run_name}"
    model_path = save_dir + "/model.pth"
    torch.save(
        {
            'model_state_dict': agent.cpu().state_dict(),
            'global_step' : global_step
        },
        model_path
    )