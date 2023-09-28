import gymnasium as gym


env = gym.make("PongNoFrameskip-v4", render_mode="human")
observation, info = env.reset(seed=42)

for idx in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    
    if idx == 0:
        print(observation.shape)
        print(reward)
        print(info)
    
    if terminated or truncated:
        print(terminated)
        print(truncated)
        observation, info = env.reset()
    
env.close()