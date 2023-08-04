import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

root_performance_logs = "../logs/performance"

def main():

    sns.set_theme(style="whitegrid", palette="pastel")
    
    num_mcts_iterations = range(10, 250, 20)

    dfs = []
    for i in num_mcts_iterations:

        x = np.load(f"{root_performance_logs}/sum_rewards_{i}.npy")

        data = {"Number of MCTS iterations": i, 
                    "Sum reward": x}

        df = pd.DataFrame(data)
        
        dfs.append(df)

    dfs = pd.concat(dfs)
    
    fig, axes = plt.subplots(2, 1, figsize=(8,8))

    sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Sum reward", ax=axes[0])
    sns.violinplot(data=dfs, x="Number of MCTS iterations", y="Sum reward", ax=axes[1])

    fig.suptitle('Vanilla MCTS: Distribution of sum rewards')
    plt.savefig("./Plots/Vanilla_MCTS_performance_sum_reward.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
