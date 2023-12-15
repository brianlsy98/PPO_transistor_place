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
    M = np.array([2, 2, 4, 10]); K = 2
    # M = np.array([2, 2, 6, 8]); K = 2
    # M = np.array([2, 2, 2, 6]); K = 1.3
    # M = np.array([6, 6, 6, 6]); K = 1.3
    # M = np.array([2, 2, 6, 8]); K = 2
    # M = np.array([4, 6, 8, 10]); K = 1.3
    # M = np.array([2, 10, 10, 10]); K = 2
    # M = np.array([6, 6, 10, 10]); K = 2
    # M = np.array([6, 6, 6, 6]); K = 1.3
    # M = np.array([10, 10, 10, 10]); K = 1.3
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
                            "D": {"G": "GCD_SB_DD", "S": "SCD", "D": "GCD_SB_DD"} }       ],
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