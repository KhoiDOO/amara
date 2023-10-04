import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def dqn_solver(envs, args, device, global_step, 
               obs, next_obs, rewards, dones, actions,
               q_network, target_network, optimizer):
    if global_step % args.train_frequency == 0:
        b_obs = obs.reshape((-1,) + envs.unwrapped.single_observation_space.shape)
        b_next_obs = next_obs.reshape((-1,) + envs.unwrapped.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.unwrapped.single_action_space.shape)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        
        with torch.no_grad():
            target_max, _ = target_network(b_next_obs).max(dim=1)
            td_target = b_rewards + args.gamma * target_max * (1 - b_dones)
        old_val = q_network(b_obs).gather(1, b_actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        # if global_step % 100 == 0:
        #     writer.add_scalar("losses/td_loss", loss, global_step)
        #     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
        #     print("SPS:", int(global_step / (time.time() - start_time)))
        #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # update target network
    if global_step % args.target_network_frequency == 0:
        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
            target_network_param.data.copy_(
                args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
            )
    
    return {
        "losses/td_loss" : loss,
        "losses/q_values" : old_val.mean().item()
    }