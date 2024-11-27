# Reinforcement-Learning-Deep-Q-Networks

This repository tackles class Reinforcement Learning problems by using Deep Q Networks implemented using PyTorch.

A DQN is a neural network designed to predict the Q function (for all the
possible actions) of the environment given a state vector. The first layer of the network takes as input the observed
state. The number of outputs in the final layer of the network must be the same number of actions the
agent can perform. 

## Cart Pole

The goal is to train an agent to balance a pole attached (by a frictionless joint) to a moving (frictionless) cart by applying a fixed force to the cart in either the left or right direction. Therefore, a DQN is trained such that it keeps the pole balanced (upright) for as many
steps as possible. The optimal policy will account for deviations from the upright position and push the cartpole such
that it remains balanced. ![Cart Pole Documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

### Example Solution Runs:

#### Run 1
![Cart Pole Balancing Act](https://github.com/aanish94/Reinforcement-Learning-Deep-Q-Networks/blob/main/results/output.gif)

#### Run 2
![Cart Pole Balancing Act](https://github.com/aanish94/Reinforcement-Learning-Deep-Q-Networks/blob/main/results/output2.gif)
