import os
from env import PlacingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import wandb

import numpy as np

def make_env(seed, netlists, cc, reward_mode, config):
    def _init():
        env = PlacingEnv(
                netlists = netlists,
                M = np.array([2, 2, 4, 4]), K = 1.3,
                general_train_mode=True, reward_mode=reward_mode,
                common_centroid=cc, config=config,
                test_netlist_i = seed%len(netlists)
            )
        env.reset()
        return env
    set_random_seed(seed)
    return _init


save_dir = "./rl/placer/models"
num_cpu = 6
if __name__ == "__main__":
        
        ######## input ########
        ccORnoncc = "noncc"
        reward_mode = 0
        gamma = 0.99
        batch_size = 2048
        learning_rate = 0.0003
        #######################

        if ccORnoncc == "cc": cc = True
        else: ccORnoncc = "noncc"; cc = False

        config = {
            "gamma": gamma,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        }

        netlists = [    {   "A": {"G": "DA", "S": "SS", "D": "DA"},
                            "B": {"G": "DA", "S": "SS", "D": "DB"},
                            "C": {"G": "DA", "S": "SS", "D": "DC"},
                            "D": {"G": "DA", "S": "SS", "D": "DD"}     },

                        {   "A": {"G": "GAB", "S": "SA_DC", "D": "DA"},
                            "B": {"G": "GAB", "S": "SB_DD", "D": "DB"},
                            "C": {"G": "GC", "S": "SCD", "D": "SA_DC"},
                            "D": {"G": "GD", "S": "SCD", "D": "SB_DD"}   },

                        {   "A": {"G": "GAB_DA", "S": "SA_DC", "D": "DA"},
                            "B": {"G": "GAB_DA", "S": "GCD_SB_DD", "D": "DB"},
                            "C": {"G": "GCD_SB_DD", "S": "SCD", "D": "SA_DC"},
                            "D": {"G": "GCD_SB_DD", "S": "SCD", "D": "GCD_SB_DD"} }       ]
        
        vec_env = SubprocVecEnv([make_env(i, netlists, cc, reward_mode, config) for i in range(num_cpu)])

        # train
        # model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")       
        model = PPO("MlpPolicy", vec_env, verbose=1, device="mps", batch_size=batch_size, gamma=gamma, learning_rate=learning_rate)


        if os.path.isfile(f'{save_dir}/{len(netlists[0])}insts/subprocvec/reward_mode_{reward_mode}/placing_agent_{ccORnoncc}.zip'):
            print(f"load placing agent with {len(netlists[0])} types of instances ({ccORnoncc})")
            model = PPO.load(f"{save_dir}/{len(netlists[0])}insts/subprocvec/reward_mode_{reward_mode}/placing_agent_{ccORnoncc}", env=vec_env)
        total = 10
        for k in range(total):
            model.learn(total_timesteps = 500000)
            model.save(f"{save_dir}/{len(netlists[0])}insts/subprocvec/reward_mode_{reward_mode}/placing_agent_{ccORnoncc}")