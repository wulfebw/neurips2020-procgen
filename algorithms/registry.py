"""
Registry of custom implemented algorithms names

Please refer to the following examples to add your custom algorithms : 

- AlphaZero : https://github.com/ray-project/ray/tree/master/rllib/contrib/alpha_zero
- bandits : https://github.com/ray-project/ray/tree/master/rllib/contrib/bandits
- maddpg : https://github.com/ray-project/ray/tree/master/rllib/contrib/maddpg
- random_agent: https://github.com/ray-project/ray/tree/master/rllib/contrib/random_agent

An example integration of the random agent is shown here : 
- https://github.com/AIcrowd/neurips2020-procgen-starter-kit/tree/master/algorithms/custom_random_agent
"""


def _import_custom_random_agent():
    from .custom_random_agent.custom_random_agent import CustomRandomAgent
    return CustomRandomAgent


def _import_random_policy():
    from .random_agent.trainer import RandomAgentTrainer
    return RandomAgentTrainer


def _import_monet_agent():
    from .monet_agent.monet_agent import MONetTrainer
    return MONetTrainer


def _import_episode_adversarial_agent():
    from .episode_adversarial_agent.episode_adversarial_agent import EpisodeAdversarialTrainer
    return EpisodeAdversarialTrainer


def _import_data_augmenting_ppo_agent():
    from .data_augmenting_ppo_agent.data_augmenting_ppo_agent import DataAugmentingPPOTrainer
    return DataAugmentingPPOTrainer


CUSTOM_ALGORITHMS = {
    "custom/CustomRandomAgent": _import_custom_random_agent,
    "custom/MONetTrainer": _import_monet_agent,
    "custom/EpisodeAdversarialTrainer": _import_episode_adversarial_agent,
    "custom/DataAugmentingPPOTrainer": _import_data_augmenting_ppo_agent,
    "RandomPolicy": _import_random_policy,
}
