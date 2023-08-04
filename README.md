# AlphaCandy-Zero
## Mastering the game of Go without Candy Crush

An AI-based on the [AlphaZero algorithm](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go) is trained to play the game of Candy Crush. The implementation of this AI is based on the [AlphaGo Zero paper](https://www.deepmind.com/publications/mastering-the-game-of-go-without-human-knowledge).

In summary, this AI is based on the Monte Carlo Tree Search (MCTS) algorithm. A neural network 


<img src="./Media/play_game_mode_0.gif" width="300" height="300">

## Usage

### Directory overview

* **./**: Files for training.
* **Media**: Contains `.png` and `.gif` files belonging to this `README.md`.
* **Plotting scripts**: Scripts creating the plots for the evaluation part of this `README.md`.
    * **Plots**: Plots created by the plotting scripts.
* **Raw MCTS**: Implementation of the vanilla MCTS algorithm including evaluation scripts. The performance is measured in form of average and sum rewards of an episode of 10 steps. Multiple MCTS algorithms instances (same number of iterations) are executed concurrently for increasing the speed of the measurement.
    * **logs**: 
        * **performance**: The averages of each launched MCTS algorithms instance is stored in a `avg_rewards_X.np` file. 
        In the `sum_rewards_X.np`the sum of rewards of each launched MCTS algorithms instance is stored.
        Note that, `X` indicates the number of iterations of the vanilla MCTS algorithm.
        * **time**: An average duration over 10 steps is determinated how long the algorithm needed for calculating an action. There is an average duration value for each concurrently executed MCTS instance. These averages are stored in a `X.npy` file.
        Same as above, `X` indicates the number of iterations of the vanilla MCTS algorithm.
* **saved_model**: A ``trained_weights_N.index`` file contains the weights of the neural network learning the policy and the value of the MCTS algorithm at each `N`.
* **test_logs**: 
    * **test**: TensorBoard log file - for each epoch this train and test loss (including for value and policy) are logged. Besides, the performance of the agent is evaluated - average and sum of rewards. These are stored in this log file.  
* **Window**:
    **Images**: Candy images (e.g normal, striped etc.) and images of the arrows which highlight the current action: top, down, left and right.
        * **Arrows**, **Normal**, **Striped** etc.


### Training

Run `Training.py` to start the training. The configuration such as number of epochs, number of iterations of the current dataset etc. can be changed in the `Config.py` file.For each epoch the weights of the neural network will be saved into `./saved_model`. 
Note that, the agent was already trained for 15 epochs. Executing `Training.py` will overwrite the stored weights in `./saved_model`.

```
python3 ./Training.py
```

### Window 

#### mode = 0

<img src="./Media/play_game_mode_0.gif" width="300" height="300">

#### mode = 1

<img src="./Media/play_game_mode_1.gif" width="1500" height="400">


## Evaluation

### Classical/vanilla MCTS  

#### Average reward

![alt text](./Raw%20MCTS/Plotting%20scripts/Plots/Vanilla_MCTS_performance_avg_reward.png)

#### Sum reward

![alt text](./Raw%20MCTS/Plotting%20scripts/Plots/Vanilla_MCTS_performance_sum_reward.png)

### Train and test loss

![alt text](./Plotting%20scripts/Plots/TrainTestLoss.png)

### Rewards

![alt text](./Plotting%20scripts/Plots/AvgSumReward.png)

#### Distribution of average rewards

![alt text](./Plotting%20scripts/Plots/DistriAvgRewards.png)

#### Distribution of sum rewards

![alt text](./Plotting%20scripts/Plots/DistriSumRewards.png)

## Requirements
- TensorFlow 2
- TensorboardX
- Numpy
- gym
- tkinter
- matplotlib
- argparse
- imageio
- pyautogui