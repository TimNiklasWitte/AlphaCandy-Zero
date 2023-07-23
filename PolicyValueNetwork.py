import tensorflow as tf

from Config import *
from CandyCrushUtiles import *

class PolicyValueNetwork(tf.keras.Model):

    def __init__(self):
     
        super(PolicyValueNetwork, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(3,3), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(3,3), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), strides=(3,3), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), strides=(3,3), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), strides=(3,3), activation="tanh", padding='same'),
            tf.keras.layers.Flatten(),
        ]

        self.policy_layer = tf.keras.layers.Dense(NUM_ACTIONS, activation="softmax")
        self.value_layer = tf.keras.layers.Dense(1)


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.policy_loss_function = tf.keras.losses.BinaryCrossentropy()
        self.value_loss_function =  tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_policy_loss = tf.keras.metrics.Mean(name="policy_loss")
        self.metric_policy_accuracy = tf.keras.metrics.Accuracy(name="accuracy")

        self.metric_value_loss = tf.keras.metrics.Mean(name="value_loss")


        # valid_actions = [isValidAction(action) for action in range(NUM_ACTIONS)]
        # self.valid_actions = tf.constant(valid_actions)

        # valid_actions = [action for action in range(NUM_ACTIONS) if not isValidAction(action)]
        # self.valid_actions = tf.constant(valid_actions)
 
    
    @tf.function
    def call(self, x):
        
        for layer in self.layer_list:
            x = layer(x)
        
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        
        # batch_size = x.shape[0]
        # policy = tf.where(self.valid_actions, policy_logits, tf.zeros(shape=(batch_size, NUM_ACTIONS, )))
        
        # #policy = tf.nn.softmax(policy_logits, axis=-1)


        return policy, value 
    

    def call_no_tf_func(self, x):
        
        for layer in self.layer_list:
            x = layer(x)
        
        policy = self.policy_layer(x)
        value = self.value_layer(x)

        return policy, value 

    @tf.function
    def train_step(self, x, target_policy, target_value):
    
        with tf.GradientTape() as tape:
            policy, value = self(x)

            policy_loss = self.policy_loss_function(target_policy, policy)
            value_loss = self.value_loss_function(target_value, value)

            loss = policy_loss + value_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.metric_loss.update_state(loss)

        self.metric_policy_loss.update_state(policy_loss)
        self.metric_value_loss.update_state(value_loss)

        prediction_action = tf.argmax(policy, axis=-1)
        target_action = tf.argmax(target_policy, axis=-1)


        self.metric_policy_accuracy.update_state(target_action, prediction_action)

    @tf.function
    def test_step(self, dataset):
        

        self.metric_loss.reset_states()

        self.metric_policy_loss.reset_states()
        self.metric_policy_accuracy.reset_states()

        self.metric_value_loss.reset_states()

        for state, target_value, target_policy in dataset:

            policy, value = self(state)

            policy_loss = self.policy_loss_function(target_policy, policy)
            value_loss = self.value_loss_function(target_value, value)

            loss = policy_loss + value_loss


            self.metric_loss.update_state(loss)
        
            self.metric_policy_loss.update_state(policy_loss)
            self.metric_value_loss.update_state(value_loss)

            prediction_action = tf.argmax(policy, axis=-1)
            target_action = tf.argmax(target_policy, axis=-1)

            self.metric_policy_accuracy.update_state(target_action, prediction_action)