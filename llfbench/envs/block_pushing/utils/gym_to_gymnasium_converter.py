import gym
import gymnasium
import numpy as np


def convert_gym_to_gymnasium_space(space):
    """
    Converts a gym.spaces object into the corresponding gymnasium.spaces object.
    :param space: A gym.spaces object (Box, Discrete, Tuple, Dict, etc.)
    :return: The equivalent gymnasium.spaces object.
    """
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(n=space.n, start=space.start if hasattr(space, 'start') else 0)
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return gymnasium.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return gymnasium.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return gymnasium.spaces.Tuple(tuple(convert_gym_to_gymnasium_space(s) for s in space.spaces))
    elif isinstance(space, gym.spaces.Dict):
        return gymnasium.spaces.Dict({
            k: convert_gym_to_gymnasium_space(v) for k, v in space.spaces.items()
        })
    else:
        raise TypeError(f"Unsupported space type: {type(space)}")


def convert_gym_to_gymnasium_env(env):
    """
    Converts a gym.Env object into a gymnasium.Env object by updating its spaces.
    :param env: A gym.Env object
    :return: An environment with gymnasium.spaces.
    """
    env.observation_space = convert_gym_to_gymnasium_space(env.observation_space)
    env.action_space = convert_gym_to_gymnasium_space(env.action_space)
    return env


# Example Usage
if __name__ == "__main__":
    # Create a gym environment
    gym_env = gym.make("CartPole-v1")
    
    # Convert spaces
    gymnasium_obs_space = convert_gym_to_gymnasium_space(gym_env.observation_space)
    gymnasium_action_space = convert_gym_to_gymnasium_space(gym_env.action_space)

    print("Original gym observation space:", gym_env.observation_space)
    print("Converted gymnasium observation space:", gymnasium_obs_space)

    # Convert entire environment
    gymnasium_env = convert_gym_to_gymnasium_env(gym_env)
    print("Environment converted successfully!")
