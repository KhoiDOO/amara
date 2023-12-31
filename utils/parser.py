import argparse
import time

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # COMMON SETTINGS
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", action="store_false",
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
    parser.add_argument("--capture-video", action="store_true",
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--log", action="store_false",
        help="Performance Logging")
    parser.add_argument("--model_save", action="store_false",
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
    parser.add_argument("--anneal-lr", action="store_false",
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--update_after_ep", action="store_false",
        help="Model Checkpoint")
    
    # ALGO SETTINGS
    parser.add_argument("--algo", type=str, default='ppo', choices = ['ppo', 'dqn'],
        help="Alogrithm used in training")
    
    # PPO SETTINGS
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", action="store_false",
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", action="store_false",
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    # DQN SETTINGS
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.10,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=80000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--buffer-size", type=int, default=1000000,
        help="the replay memory buffer size")
    
    args = parser.parse_args()
    if args.algo == "ppo":
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.run_name = f"{args.env_id}__{args.algo}__{args.seed}__{int(time.time())}"
    
    return args