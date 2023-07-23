import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

root_performance_logs = "./logs/performance"

def main():

    sns.set_theme(style="ticks", palette="pastel")
    
    num_mcts_iterations = range(50, 500, 50)

    dfs = []
    for i in num_mcts_iterations:

        x = np.load(f"{root_performance_logs}/avg_rewards_{i}.npy")

        data = {"Number of MCTS iterations": i, 
                    "Average reward": x}

        df = pd.DataFrame(data)

        dfs.append(df)

    dfs = pd.concat(dfs)
    print(dfs)
    sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Average reward")
    sns.despine(offset=10, trim=True)

    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
