import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



def main():

    sns.set_theme(style="whitegrid", palette="pastel")
    
    num_mcts_iterations = range(25, 275, 25)

    dfs = []

    for candy_buff_height in [1, 10, 15]:

        root_performance_logs = f"../logs/candy_buff_height_{candy_buff_height}/performance/"

        for num_mcts_iteration in num_mcts_iterations:

            x = np.load(f"{root_performance_logs}/sum_rewards_{num_mcts_iteration}.npy")

            data = {"Number of MCTS iterations": num_mcts_iteration, 
                    "Candy buff height": candy_buff_height,
                        "Sum reward": x}

            df = pd.DataFrame(data)
            
            dfs.append(df)

    dfs = pd.concat(dfs)
    
    # print(dfs)
    # return 
    #fig, axes = plt.subplots(2, 1, figsize=(8,8))

    sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Sum reward", hue="Candy buff height")
    #sns.violinplot(data=dfs, x="Number of MCTS iterations", y="Sum reward", ax=axes[1])

    #fig.suptitle('Vanilla MCTS: Distribution of sum rewards')
    #plt.savefig("./Plots/Vanilla_MCTS_performance_sum_reward.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
