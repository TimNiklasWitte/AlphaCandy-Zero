import tensorflow as tf
import numpy as np
import tqdm

from MCTS import *
from MCTS_Buffer import *
from PolicyValueNetwork import *
from Config import *
from Evaluation import *
from Logging import *

from matplotlib import pyplot as plt

# openai gym causes a warning - disable it
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')


def main():
  
    #
    # MCTS_Buffer: Its data is used by a TensorFlow Dataset object
    #
    reduced_action_space = get_reduced_action_space()
    len_reduced_action_space = len(reduced_action_space)
    mcts_buffer = MCTS_Buffer(MCTS_BUFFER_SIZE, len_reduced_action_space, STATE_SHAPE)

    #
    # Logging
    # 
    train_summary_writer = tf.summary.create_file_writer(logs_file_path)


    #
    # Initialize model
    #

    policyValueNetwork = PolicyValueNetwork(num_actions=len_reduced_action_space)
    policyValueNetwork.build(input_shape=(1,*STATE_IMG_SHAPE))

    policyValueNetwork.summary()
    
 
    update_dataset(mcts_buffer, policyValueNetwork)

    return 
    dataset = tf.data.Dataset.from_generator(
                    mcts_buffer.dataset_generator,
                    output_signature=(
                            tf.TensorSpec(shape=STATE_IMG_SHAPE , dtype=STATE_IMG_DTYPE),
                            tf.TensorSpec(shape=(1,), dtype=VALUE_DTYPE),
                            tf.TensorSpec(shape=(len_reduced_action_space), dtype=POLICY_DTYPE)
                        )
                )
    

    # Train & Test split
    dataset_size = mcts_buffer.num_samples
    TEST_DATASET_SIZE = int((dataset_size / 100) * TEST_DATASET_SIZE_PERCENTAGE)
    test_dataset = dataset.take(TEST_DATASET_SIZE) 
    train_dataset = dataset.skip(TEST_DATASET_SIZE)

    # print(dataset_size)
    # print(len(list(train_dataset)))
    # print(len(list(test_dataset)))

    # Proprocessing Pipeline
    train_dataset = train_dataset.apply(prepare_data)
    test_dataset = test_dataset.apply(prepare_data)

     
    # for state, target_value, target_policy in train_dataset:
        
    #     # policy, value = policyValueNetwork(state)

    #     # print(policy)
    #     # policy = tf.reduce_sum(policy, axis=-1)
    #     # print(policy)

    #     state = state[0]

    #     plt.imshow(state)
    #     plt.show()

    #     print(state)
    #     print(target_value)
    #     print(target_policy)
    #     break

    # return

    print("<INFO> Epoch: 0")
    log(train_summary_writer, policyValueNetwork, train_dataset, test_dataset, 0)


    #
    # Print initial setup
    #

    # Model
    policyValueNetwork.summary()

 
    for epoch in range(1, NUM_EPOCHS):
        
        print(f"<INFO> Epoch: {epoch}")
        if epoch != 1:
            update_dataset(mcts_buffer, policyValueNetwork)

            dataset = tf.data.Dataset.from_generator(
                    mcts_buffer.dataset_generator,
                    output_signature=(
                            tf.TensorSpec(shape=STATE_IMG_SHAPE , dtype=STATE_IMG_DTYPE),
                            tf.TensorSpec(shape=(), dtype=VALUE_DTYPE),
                            tf.TensorSpec(shape=(), dtype=ACTION_DTYPE)
                        )
                )
            
            dataset_size = mcts_buffer.num_samples
            TEST_DATASET_SIZE = int((dataset_size / 100) * TEST_DATASET_SIZE_PERCENTAGE)
            test_dataset = dataset.take(TEST_DATASET_SIZE) 
            train_dataset = dataset.skip(TEST_DATASET_SIZE)

            train_dataset = train_dataset.apply(prepare_data)
            test_dataset = test_dataset.apply(prepare_data)



        for num_train_loop in range(NUM_TRAIN_LOOPS):
            print(f"<INFO> Train loop: {num_train_loop}/{NUM_TRAIN_LOOPS}")
            for state, target_value, target_policy in tqdm.tqdm(train_dataset,position=0, leave=True):
                policyValueNetwork.train_step(state, target_policy, target_value)

   
        log(train_summary_writer, policyValueNetwork, train_dataset, test_dataset, epoch)
        
        policyValueNetwork.save_weights(f"./saved_model/trained_weights_{epoch}", save_format="tf")
    # Free -> Prevent memory leak (warning otherwise)
    mcts_buffer.free_shms()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")