# AlphaCandy-Zero
##

An AI-based on the [AlphaZero algorithm](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go) is trained to play the game of Candy Crush. The implementation of this AI is based on the [AlphaGo Zero paper](https://www.deepmind.com/publications/mastering-the-game-of-go-without-human-knowledge).


<img src="./Media/play_game_mode_0.gif" width="300" height="300">



## Usage

### Training



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