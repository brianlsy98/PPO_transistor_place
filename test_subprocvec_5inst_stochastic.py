import os
from env import PlacingEnv
from stable_baselines3 import PPO

import numpy as np

save_dir = "./rl/placer/models"

if __name__ == "__main__":

    ######## input ########
    ccORnoncc = "cc"
    reward_mode = 0
    gamma = 0.99
    batch_size = 2048
    learning_rate = 0.0003

    netlist = 0
    M = np.array([4,4,4,10,10]); K = 2
    # M = np.array([4, 4, 4, 6, 6]); K = 1.3
    # M = np.array([2, 2, 6, 6, 2]); K = 2
    # M = np.array([4, 6, 10, 2, 2]); K = 1.3
    # M = np.array([4, 6, 10, 2, 2]); K = 1.3
    # M = np.array([2, 2, 4, 8, 8]); K = 1.3
    # M = np.array([4, 4, 4, 10, 10]); K = 2
    # M = np.array([4, 6, 8, 10, 10]); K = 2
    # M = np.array([4, 4, 4, 6, 6]); K=1.3
    # M = np.array([4, 6, 6, 6, 6]); K=1.3
    # M = np.array([2, 4, 4, 4, 4]); K=2
    # M = np.array([6, 6, 6, 6, 8]); K=2
    #######################

    if ccORnoncc == "cc": cc = True
    else: ccORnoncc = "noncc"; cc = False

    config = {
        "gamma": gamma,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }

    # Generate placing Env with the above informations
    placing_env = PlacingEnv(
        netlists = [    {   "M0": {"G": "DA", "S": "SS", "D": "DA"},
                            "M1": {"G": "DA", "S": "SS", "D": "DB"},
                            "M2": {"G": "DA", "S": "SS", "D": "DC"},
                            "M3": {"G": "DA", "S": "SS", "D": "DD"},
                            "M4": {"G": "DA", "S": "SS", "D": "DE"}     },

                        {   "M0": {"G": "GA_DB", "S": "SA_DC", "D": "GB_DA"},
                            "M1": {"G": "GB_DA", "S": "SB_DD", "D": "GA_DB"},
                            "M2": {"G": "GC", "S": "SCD_DE", "D": "SA_DC"},
                            "M3": {"G": "GD", "S": "SCD_DE", "D": "SB_DD"},
                            "M4": {"G": "GE", "S": "SE", "D": "SCD_DE"}   },

                        {   "M0": {"G": "GABC_DA", "S": "SABC", "D": "GABC_DA"},
                            "M1": {"G": "GABC_DA", "S": "SABC", "D": "DB_SDE"},
                            "M2": {"G": "GABC_DA", "S": "SABC", "D": "DC"},
                            "M3": {"G": "GD", "S": "DB_SDE", "D": "DD"},
                            "M4": {"G": "GE", "S": "DB_SDE", "D": "DE"}   }       ],
        test_netlist_i = netlist,
        M = M, K = K,
        general_train_mode=False,
        reward_mode=reward_mode, common_centroid=cc,
        config=config
    )

    model = PPO("MlpPolicy", placing_env, verbose=0, device="mps", gamma=gamma, batch_size=batch_size, learning_rate=learning_rate)

    # test
    if os.path.isfile(f'{save_dir}/{len(placing_env.netlist)}insts/subprocvec/reward_mode_{reward_mode}/placing_agent_{ccORnoncc}.zip'):
        print()
        print(f"... load trained placing_agent_cc.zip ...")
        model = PPO.load(f"{save_dir}/{len(placing_env.netlist)}insts/subprocvec/reward_mode_{reward_mode}/placing_agent_{ccORnoncc}", env=placing_env)

    sample_n = 100
    for i in range(sample_n):
        obs = placing_env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation=obs, deterministic=False)
            obs, rewards, done, info = placing_env.step(action)
            if done :
                placing_env.render()
                if i == 0 or placing_env.total_reward > best_info["reward"]:
                    best_info = info


    print()
    print("CC : ", cc)
    print()
    print(f"M={M}, K={K}, netlist={netlist}")
    print()
    for key, value in best_info.items():
        print(f"{key} :\n{value}\n")

    print(f"centroid dist sum      : {np.sum(best_info['centroid_dists'])}")
    print(f"dispersion degree var  : {np.var(best_info['dispersion_degrees'])}")
    print(f"LOD var                : {np.var(best_info['LOD'])}")
    print(f"routing complexity sum : {np.sum(best_info['routing_complexity'])}")
    print()