import argparse
import time

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # COMMON SETTINGS
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=bool, default=True,
        help="Setup torch.backends.cudnn.deterministic")
    parser.add_argument("--verbose", action='store_true', 
        help="Toggle to print reward and episodic length every end of episode")
    parser.add_argument("--dvidx", type=int, default=0,
        help="Index of the device used in training")
    parser.add_argument("--problem", type=str, default="gym_atari", choices = ["gym_atari"])
        
    # LOGGING SETTINGS
    parser.add_argument("--wandb-project-name", type=str, default="RL-baselines",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="scalemind",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=bool, default=False,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--log", type=bool, default=True,
        help="Performance Logging")
    parser.add_argument("--model_save", type=bool, default=True,
        help="Model Checkpoint")

    # TRAINING SETTINGS
    parser.add_argument("--env-id", type=str, default="PongNoFrameskip-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--opt", type=str, default='adam', choices = ['adam'],
        help="optimizer used in training")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=bool, default=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--update_after_ep", type=bool, default=True,
        help="Model Checkpoint")
    
    # ALGO SETTINGS
    parser.add_argument("--algo", type=str, default='ppo', choices = ['ppo'],
        help="Alogrithm used in training")
    
    # PPO SETTINGS
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=bool, default=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=bool, default=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.run_name = f"{args.env_id}__{args.algo}__{args.seed}__{int(time.time())}"
    
    return args