import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

path = "./logs/time/log.npy"

NUM_STEPS = 50
def main():

    durations = np.load(path)

    print(durations.shape)
    #durations = pd.DataFrame(durations)
    #durations.columns = ["Duration (ms)"]
    #durations.index.name = "Number of MCTS iterations"
    
    dfs = []
    for i in range(50, 500, 50):
       
        data = {"Number of MCTS iterations": i, 
                    "Duration": durations[i, :]}

        df = pd.DataFrame(data)

        dfs.append(df)

    dfs = pd.concat(dfs)
    print(dfs)
    sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Duration")
    sns.despine(offset=10, trim=True)

    plt.show()
    # for i in range(NUM_STEPS):

    # print(durations)

    # sns.lineplot(data=durations, x="Number of MCTS iterations", y="Duration (ms)")

    # plt.show()

    # sns.set_theme(style="ticks", palette="pastel")
    
    # num_mcts_iterations = range(10, 260, 10)

    # dfs = []
    # for i in num_mcts_iterations:

    #     x = np.load(f"{root_logs}/{i}.npy")

    #     data = {"Number of MCTS iterations": i, 
    #                 "Number of steps": x}

    #     df = pd.DataFrame(data)

    #     dfs.append(df)

    # dfs = pd.concat(dfs)

    # sns.boxplot(data=dfs, x="Number of MCTS iterations", y="Number of steps")
    # sns.despine(offset=10, trim=True)

    # plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
