import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from LoadDataframe import *

def main():
    
    df = load_dataframe_tensors()
    print(df.columns)
    #
    # Plotting
    #

    fig, axes = plt.subplots(1, 2, figsize=(10,5))

    sns.lineplot(data=df.loc[:, ["avg reward", "avg reward network only"]], ax=axes[0], palette=['r', 'g']).set(title='Average reward')
    sns.lineplot(data=df.loc[:, ["sum reward", "sum reward network only"]], ax=axes[1], palette=['r', 'g']).set(title='Sum reward')

    fig.suptitle('Rewards')
    fig.tight_layout()

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.savefig("./Plots/AvgSumReward.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")