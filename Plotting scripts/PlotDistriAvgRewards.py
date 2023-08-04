import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from LoadDataframe import *

def main():
    
    df = load_dataframe_text()
    
    mask = (df["tag"] == "avg rewards") | (df["tag"] == "avg rewards network only")
    df = df.loc[mask, ["tag","Epoch","value"]]
   
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(8,8))

    ax = sns.boxplot(data=df, x="Epoch", y="value", hue="tag", ax=axes[0])
    ax.legend_.set_title(None)
    ax.set(ylabel=None)

    ax = sns.violinplot(data=df, x="Epoch", y="value", hue="tag", split=True, ax=axes[1])
    ax.legend_.set_title(None)
    ax.set(ylabel=None)
   
    fig.suptitle('Distribution of average rewards')
    fig.tight_layout()
    plt.savefig("./Plots/DistriAvgRewards.png")
    plt.show()
    

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")