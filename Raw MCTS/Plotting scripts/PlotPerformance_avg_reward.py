import numpy as np

from matplotlib import pyplot as plt
from matplotlib import rcParams
import pandas as pd
import seaborn as sns

root_performance_logs = "../logs/performance"

def main():

    sns.set_theme(style="ticks", palette="pastel")
    
    num_mcts_iterations = range(10, 250, 20)

    dfs = []
    for i in num_mcts_iterations:

        x = np.load(f"{root_performance_logs}/avg_rewards_{i}.npy")

        data = {"Number of MCTS iterations": i, 
                    "Average reward": x}

        df = pd.DataFrame(data)
        
        dfs.append(df)

    dfs = pd.concat(dfs)
    
    rcParams['figure.figsize'] = 7, 6
    ax = sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Average reward")
    ax.set(title='Average reward of vanilla MCTS')
    sns.despine(offset=10, trim=True)
    
    
    plt.savefig("./Plots/Vanilla_MCTS_performance_avg_reward.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
