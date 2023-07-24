import tensorflow as tf
import numpy as np 

from Evaluation import *


def log(train_summary_writer, policyValueNetwork, train_dataset, test_dataset, epoch):

    # Epoch 0 = no training steps are performed 
    # test based on train data
    # -> Determinate initial train_loss and train_accuracy
    if epoch == 0:
        policyValueNetwork.test_step(train_dataset.take(500))

    #
    # Train
    #
    train_loss = policyValueNetwork.metric_loss.result()

    train_policy_loss = policyValueNetwork.metric_policy_loss.result()
    train_policy_accuracy = policyValueNetwork.metric_policy_accuracy.result()

    train_value_loss = policyValueNetwork.metric_value_loss.result()


    # Reset metrices
    policyValueNetwork.metric_loss.reset_states()

    policyValueNetwork.metric_policy_loss.reset_states()
    policyValueNetwork.metric_policy_accuracy.reset_states()

    policyValueNetwork.metric_value_loss.reset_states()

    #
    # Test
    #

    policyValueNetwork.test_step(test_dataset)

    test_loss = policyValueNetwork.metric_loss.result()

    test_policy_loss = policyValueNetwork.metric_policy_loss.result()
    test_policy_accuracy = policyValueNetwork.metric_policy_accuracy.result()

    test_value_loss = policyValueNetwork.metric_value_loss.result()

    # Reset metrices
    policyValueNetwork.metric_loss.reset_states()

    policyValueNetwork.metric_policy_loss.reset_states()
    policyValueNetwork.metric_policy_accuracy.reset_states()

    policyValueNetwork.metric_value_loss.reset_states()

    #
    # use_policy_network_only=True
    #
    avg_rewards_mem_network_only, sum_rewards_mem_network_only = evaluate(policyValueNetwork, use_policy_network_only=True)

    avg_reward_network_only = np.average(avg_rewards_mem_network_only)
    sum_reward_network_only = np.average(sum_rewards_mem_network_only)

    avg_rewards_network_only = np.array2string(avg_rewards_mem_network_only)
    sum_rewards_network_only = np.array2string(sum_rewards_mem_network_only)
    
    #
    # use_policy_network_only=False
    #
    avg_rewards_mem, sum_rewards_mem = evaluate(policyValueNetwork, use_policy_network_only=False)

    avg_reward = np.average(avg_rewards_mem)
    sum_reward = np.average(sum_rewards_mem)

    avg_rewards = np.array2string(avg_rewards_mem)
    sum_rewards = np.array2string(sum_rewards_mem)

    #
    # Write to TensorBoard
    #
    with train_summary_writer.as_default():
        tf.summary.scalar("train_loss", train_loss, step=epoch)
        tf.summary.scalar("train_policy_loss", train_policy_loss, step=epoch)
        tf.summary.scalar("train_policy_accuracy", train_policy_accuracy, step=epoch)
        tf.summary.scalar("train_value_loss", train_value_loss, step=epoch)

        tf.summary.scalar("test_loss", test_loss, step=epoch)
        tf.summary.scalar("test_policy_loss", test_policy_loss, step=epoch)
        tf.summary.scalar("test_policy_accuracy", test_policy_accuracy, step=epoch)
        tf.summary.scalar("test_value_loss", test_value_loss, step=epoch)

        tf.summary.scalar("avg_reward_network_only", avg_reward_network_only, step=epoch)
        tf.summary.scalar("sum_reward_network_only", sum_reward_network_only, step=epoch)

        tf.summary.text("avg_rewards_network_only", avg_rewards_network_only, step=epoch)
        tf.summary.text("sum_rewards_network_only", sum_rewards_network_only, step=epoch)


        tf.summary.scalar("avg_reward", avg_reward, step=epoch)
        tf.summary.scalar("sum_reward", sum_reward, step=epoch)

        tf.summary.text("avg_rewards", avg_rewards, step=epoch)
        tf.summary.text("sum_rewards", sum_rewards, step=epoch)

    #
    # Output
    #
    print(f"             train_loss: {train_loss:9.4f}")
    print(f"              test_loss: {test_loss:9.4f}")
    print(f"      train_policy_loss: {train_policy_loss:9.4f}")
    print(f"       test_policy_loss: {test_policy_loss:9.4f}")
    print(f"       train_value_loss: {train_value_loss:9.4f}")
    print(f"        test_value_loss: {test_value_loss:9.4f}")
    print()
    print(f"  train_policy_accuracy: {train_policy_accuracy:9.4f}")
    print(f"   test_policy_accuracy: {test_policy_accuracy:9.4f}")
    print()
    print(f"avg_reward_network_only: {avg_reward_network_only:9.4f}")
    print(f"sum_reward_network_only: {sum_reward_network_only:9.4f}")
    print()
    print(f"             avg_reward: {avg_reward:9.4f}")
    print(f"             sum_reward: {sum_reward:9.4f}")
    print()