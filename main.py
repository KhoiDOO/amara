import os
import random
import time
from tqdm import tqdm

import torch
from torch.optim import *
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import wandb

from utils import *
from model import *
from envir import *
from solver import *
from wrapper import *

from stable_baselines3.common.buffers import ReplayBuffer

model_map = {
    "nano_cnn_ppo_agent" : Nano_CNN_PPO_Agent,
    "nano_cnn_dqn_agent" : Nano_CNN_DQN_Agent
}

env_map = {
    "gym_atari_env" : gym_atari_make_env
}

opt_map = {
    "adam" : Adam 
}

solver_map = {
    "ppo" : ppo_solver,
    "dqn" : dqn_solver
}

if __name__ == "__main__":
    args = parse_args()
    
    # Logging setup
    if args.log:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index = args.dvidx)
    
    # Environment Setup
    envs = env_map[f"{args.problem}_env"](args)
    
    # Agent Setup
    if args.algo == "ppo":
        agent = model_map[f"nano_cnn_{args.algo}_agent"](envs).to(device)
        
        opt = opt_map[args.opt](agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif args.algo == "dqn":
        agent = model_map[f"nano_cnn_{args.algo}_agent"](envs).to(device)
        opt = opt_map[args.opt](agent.parameters(), lr=args.learning_rate)
        target_network = model_map[f"nano_cnn_{args.algo}_agent"](envs).to(device)
        target_network.load_state_dict(agent.state_dict())
    
    if args.log:
        wandb.watch(models=agent, log="all")
    
    # Solver Setup
    solver = solver_map[f"{args.algo}"]
    
    # Storage Setup
    if args.algo == "ppo":
        obs_storage = torch.zeros((args.num_steps, args.num_envs) + envs.unwrapped.single_observation_space.shape).to(device)
        actions_storage = torch.zeros((args.num_steps, args.num_envs) + envs.unwrapped.single_action_space.shape).to(device)
        rewards_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs_storage = torch.zeros((args.num_steps, args.num_envs)).to(device)
    elif args.algo == "dqn":
        storage = ReplayBuffer(
            args.buffer_size, envs.unwrapped.single_observation_space, envs.unwrapped.single_action_space,
            device, optimize_memory_usage=True, handle_timeout_termination=False,
        )
    
    # Training
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    pbar = range(1, num_updates + 1) if args.verbose else tqdm(range(1, num_updates + 1))
    
    for update in pbar:
        if args.anneal_lr and args.algo != "dqn":
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            opt.param_groups[0]["lr"] = lrnow
            
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs_storage[step] = next_obs
            dones_storage[step] = next_done

            with torch.no_grad():
                if args.algo == "ppo":
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values_storage[step] = value.flatten()
                elif args.algo == "dqn":
                    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
                    if random.random() < epsilon:
                        action = np.array([envs.unwrapped.single_action_space.sample() for _ in range(envs.num_envs)])
                    else:
                        q_values = agent(torch.Tensor(next_obs).to(device))
                        action = torch.argmax(q_values, dim=1).cpu().numpy()
            
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            actions_storage[step] = action
            
            if args.algo == "ppo":
                logprobs_storage[step] = logprob

            old_obs = next_obs.cpu().numpy()
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards_storage[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            if args.algo == "dqn":
                next_obs_storage[step] = next_obs
            
            if "episode" in info:
                
                rec_lst = [x for x in info['episode']['r'].tolist() if x != 0]
                len_lst = [x for x in info['episode']['l'].tolist() if x != 0]
                
                if len(rec_lst) > 0 and len(len_lst) > 0:
                    _rec = max(rec_lst)
                    _len = max(len_lst)
                    
                    # for idx, (_rec, _len) in enumerate(zip(ep_ret, ep_len)):
                    if args.verbose:
                        print(f"gstep={global_step}, ep_r={_rec}, ep_l={_len}")
                    if args.log:
                        writer.add_scalar("charts/episodic_return", _rec, global_step)
                        wandb.log({"charts/episodic_return": _rec}, step=global_step)
                    if args.log:
                        writer.add_scalar("charts/episodic_length", _len, global_step)
                        wandb.log({"charts/episodic_length": _len}, step=global_step)
                    if args.update_after_ep:
                        break
        
        # update model
        common_dict = {
            "envs" : envs, "args" : args, "device" : device
        }
        if args.algo == "ppo":
            buffer_dict = {
                "obs" : obs_storage, "next_obs" : next_obs, "actions" : actions_storage, "logprobs" : logprobs_storage, 
                "rewards" : rewards_storage, "dones" : dones_storage, "next_done" : next_done, "values" : values_storage
            }
        elif args.algo == "dqn":
            buffer_dict = {
                "obs" : obs_storage, "next_obs" : next_obs_storage, "rewards" : rewards_storage, 
                "dones" : dones_storage, "actions" : actions_storage, "global_step" : global_step
            }
        
        if args.algo == "ppo":
            param_dict = {"agent" : agent, "optimizer" : opt}
        elif args.algo == "dqn":
            param_dict = {"q_network" : agent, "target_network" : target_network, "optimizer" : opt}
            
        buffer_dict.update(common_dict)
        param_dict.update(buffer_dict)
        
        if args.algo == "ppo":
            eval_dict = solver(**param_dict)
        elif args.algo == "dqn" and global_step > args.learning_starts:
            eval_dict = solver(**param_dict)
        
        if args.log:
            writer.add_scalar("charts/learning_rate", opt.param_groups[0]["lr"], global_step)
            wandb.log({"charts/learning_rate" : opt.param_groups[0]["lr"]}, step=global_step)
            
            writer.add_scalar("chartsfrom stable_baselines3.common.buffers import ReplayBuffer/SPS", int(global_step / (time.time() - start_time)), global_step)
            wandb.log({"charts/SPS" : int(global_step / (time.time() - start_time))}, step=global_step)
            
            for key in eval_dict:
                writer.add_scalar(key, eval_dict[key], global_step)
                wandb.log({key : eval_dict[key]}, step=global_step)
    
    envs.close()
    writer.close()
    
    if args.model_save:
        save_dir = os.getcwd() + f"/runs/{args.run_name}"
        model_path = save_dir + "/model.pth"
        torch.save(
            {
                'model_state_dict': agent.cpu().state_dict(),
                'global_step' : global_step
            },
            model_path
        )