import tensorflow as tf
import numpy as np

from Config import *
from CandyCrushUtiles import *

class PolicyValueNetwork(tf.keras.Model):

    def __init__(self, num_actions):
     
        super(PolicyValueNetwork, self).__init__()

    
        self.frontend_layer_list = [
            tf.keras.layers.Conv2D(filters=40, kernel_size=(4,4), strides=(4,4), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=30, kernel_size=(4,4), strides=(4,4), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=15, kernel_size=(4,4), strides=(4,4), activation="tanh", padding='same'),
            tf.keras.layers.Conv2D(filters=8, kernel_size=(4,4), strides=(4,4), activation="tanh", padding='same'),
            tf.keras.layers.Flatten(),
        ]

        self.backend_policy_layer_list = [
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.Dense(num_actions, activation="softmax")
        ]

        self.backend_value_layer_list = [
            tf.keras.layers.Dense(10, activation="tanh"),
            tf.keras.layers.Dense(1)
        ]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.policy_loss_function = tf.keras.losses.CategoricalCrossentropy()
        self.value_loss_function =  tf.keras.losses.MeanSquaredError()

        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_policy_loss = tf.keras.metrics.Mean(name="policy_loss")
        self.metric_value_loss = tf.keras.metrics.Mean(name="value_loss")

    
    @tf.function
    def call(self, x):
        
        for layer in self.frontend_layer_list:
            x = layer(x)
        
        
        state_embedding = x


        x = self.backend_policy_layer_list[0](state_embedding)
        for layer in self.backend_policy_layer_list[1:]:
            x = layer(x)

        policy = x 

        x = self.backend_value_layer_list[0](state_embedding)
        for layer in self.backend_value_layer_list[1:]:
            x = layer(x)
        
        value = x

        return policy, value 
    

    def call_no_tf_func(self, x):
        
        for layer in self.frontend_layer_list:
            x = layer(x)
        
        
        state_embedding = x


        x = self.backend_policy_layer_list[0](state_embedding)
        for layer in self.backend_policy_layer_list[1:]:
            x = layer(x)

        policy = x 

        x = self.backend_value_layer_list[0](state_embedding)
        for layer in self.backend_value_layer_list[1:]:
            x = layer(x)
        
        value = x

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

    @tf.function
    def test_step(self, dataset):
        

        self.metric_loss.reset_states()

        self.metric_policy_loss.reset_states()
        self.metric_value_loss.reset_states()

        for state, target_value, target_policy in dataset:

            policy, value = self(state)

            policy_loss = self.policy_loss_function(target_policy, policy)
            value_loss = self.value_loss_function(target_value, value)

            loss = policy_loss + value_loss


            self.metric_loss.update_state(loss)
        
            self.metric_policy_loss.update_state(policy_loss)
            self.metric_value_loss.update_state(value_loss)