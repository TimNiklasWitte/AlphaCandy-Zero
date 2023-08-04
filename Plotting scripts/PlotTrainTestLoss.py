import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from LoadDataframe import *

def main():

    df = load_dataframe_tensors()

    fig, axes = plt.subplots(1, 3, figsize=(8,5))

    sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], ax=axes[0])
    sns.lineplot(data=df.loc[:, ["train policy loss", "test policy loss"]], ax=axes[1])
    sns.lineplot(data=df.loc[:, ["train value loss", "test value loss"]], ax=axes[2])

    fig.suptitle('Train and test loss')
    fig.tight_layout()

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.savefig("./Plots/TrainTestLoss.png")
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")