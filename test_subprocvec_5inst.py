import os
from env import PlacingEnv
from stable_baselines3 import PPO

import numpy as np

save_dir = "./rl/placer/models"

if __name__ == "__main__":

    cc = True
    # cc = False

    # Generate placing Env with the above informations
    placing_env = PlacingEnv(
        netlists = [    {   "M0": {"G": "DA", "S": "SS", "D": "DA"},
                            "M1": {"G": "DA", "S": "SS", "D": "DB"},
                            "M2": {"G": "DA", "S": "SS", "D": "DC"},
                            "M3": {"G": "DA", "S": "SS", "D": "DD"},
                            "M4": {"G": "DA", "S": "SS", "D": "DE"}     },

                        {   "M0": {"G": "GAB", "S": "SA_DC", "D": "DA"},
                            "M1": {"G": "GAB", "S": "SB_DD", "D": "DB"},
                            "M2": {"G": "GC", "S": "SCD_DE", "D": "SA_DC"},
                            "M3": {"G": "GD", "S": "SCD_DE", "D": "SB_DD"},
                            "M4": {"G": "GE", "S": "SE", "D": "SCD_DE"}   },

                        {   "M0": {"G": "GA", "S": "SA_DBE", "D": "DA"},
                            "M1": {"G": "GB", "S": "SB", "D": "SA_DBE"},
                            "M2": {"G": "GC", "S": "SCE_DD", "D": "DC"},
                            "M3": {"G": "GD", "S": "SD", "D": "SCE_DD"},
                            "M4": {"G": "GE", "S": "SCE_DD", "D": "SA_DBE"} }       ],
        test_netlist_i = 0,
        # M = np.array([2, 2, 4, 8, 8   ]), K = 1.3,
        # M = np.array([4, 4, 4, 10, 10]), K = 2,
        # M = np.array([4, 6, 8, 10, 10]), K = 2,
        M = np.array([4, 6, 6, 6, 6]), K=1.3,
        general_train_mode=False, reward_mode=0, common_centroid=cc
    )

    # test
    if cc:
        total = 5
        for k in range(total):
            # model = PPO("MlpPolicy", placing_env, verbose=1, device="mps")
            model = PPO("MlpPolicy", placing_env, verbose=0, device="cpu")
            if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_cc_checkpoint_{k}.zip'):
                print()
                print(f"... load trained placing_agent_cc checkpoint {k} ...")
                model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_cc_checkpoint_{k}", env=placing_env)

            obs = placing_env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation=obs, deterministic=True)
                obs, rewards, done, info = placing_env.step(action)
                if done :
                    print("checkpoint ", k)
                    placing_env.render()
                
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
        total = 5
        for k in range(total):
            # model = PPO("MlpPolicy", placing_env, verbose=1, device="mps")
            model = PPO("MlpPolicy", placing_env, verbose=0, device="cpu")
            if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc_checkpoint_{k}.zip'):
                print()
                print(f"... load trained placing_agent_noncc checkpoint {k} ...")
                model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc_checkpoint_{k}", env=placing_env)

            obs = placing_env.reset()
            done = False
            while not done:
                action, _ = model.predict(observation=obs, deterministic=True)
                obs, rewards, done, info = placing_env.step(action)
                if done :
                    print("checkpoint ", k)
                    placing_env.render()
                
        model = PPO("MlpPolicy", placing_env, verbose=0, device="cpu")
        if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc_1.zip'):
            print()
            print(f"... load trained placing_agent_noncc.zip ...")
            model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/placing_agent_noncc_1", env=placing_env)

        obs = placing_env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=True)
            obs, rewards, done, info = placing_env.step(action)
            if done :
                placing_env.render()
