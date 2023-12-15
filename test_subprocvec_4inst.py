import os
from env import PlacingEnv
from stable_baselines3 import PPO

import numpy as np

save_dir = "./rl/placer/models"

if __name__ == "__main__":

    # cc = False
    cc = True

    # Generate placing Env with the above informations
    placing_env = PlacingEnv(
        netlists = [    {   "M0": {"G": "DA", "S": "SS", "D": "DA"},
                            "M1": {"G": "DA", "S": "SS", "D": "DB"},
                            "M2": {"G": "DA", "S": "SS", "D": "DC"},
                            "M3": {"G": "DA", "S": "SS", "D": "DD"}     },

                        {   "M0": {"G": "GAB", "S": "SA_DC", "D": "DA"},
                            "M1": {"G": "GAB", "S": "SB_DD", "D": "DB"},
                            "M2": {"G": "GC", "S": "SCD", "D": "SA_DC"},
                            "M3": {"G": "GD", "S": "SCD", "D": "SB_DD"}   },

                        {   "M0": {"G": "GAB_SA_DC", "S": "GAB_SA_DC", "D": "DA"},
                            "M1": {"G": "GAB_SA_DC", "S": "GCD_SB_DD", "D": "DB"},
                            "M2": {"G": "GCD_SB_DD", "S": "SC", "D": "GAB_SA_DC"},
                            "M3": {"G": "GCD_SB_DD", "S": "SD", "D": "GCD_SB_DD"} }       ],
        test_netlist_i = 0,
        # M = np.array([2, 2, 4, 10]), K = 2,
        M = np.array([2, 2, 4, 8]), K = 1.3,
        # M = np.array([4, 4, 8, 8]), K = 1.3,
        # M = np.array([2, 10, 10, 10]), K = 1.3,
        # M = np.array([2, 2, 4, 6]), K = 1.3,
        # M = np.array([2, 2, 2, 6]), K = 1.3,
        general_train_mode=False, reward_mode=0, common_centroid=cc
    )

    # test
    if cc:

        model = PPO("MlpPolicy", placing_env, verbose=0, device="cpu")
        if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_cc.zip'):
            print()
            print(f"... load trained placing_agent_cc.zip ...")
            model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_cc", env=placing_env)

        obs = placing_env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, rewards, done, info = placing_env.step(action)
            if done :
                placing_env.render()

    if not cc:
                
        model = PPO("MlpPolicy", placing_env, verbose=0, device="cpu")
        if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc.zip'):
            print()
            print(f"... load trained placing_agent_noncc.zip ...")
            model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc", env=placing_env)

        obs = placing_env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, rewards, done, info = placing_env.step(action)
            if done :
                placing_env.render()
