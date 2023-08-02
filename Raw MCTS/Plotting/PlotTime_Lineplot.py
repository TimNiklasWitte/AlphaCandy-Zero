import numpy as np

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

path = "./logs/time/log.npy"

NUM_STEPS = 50
def main():

    durations = np.load(path)
    durations = np.average(durations, axis=1)

    durations = pd.DataFrame(durations)
    durations.columns = ["Duration (ms)"]
    durations.index.name = "Number of MCTS iterations"
    
    sns.lineplot(data=durations, x="Number of MCTS iterations", y="Duration (ms)")

    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")
