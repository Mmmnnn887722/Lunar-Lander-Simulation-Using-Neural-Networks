# Lunar-Lander-Simulation-Using-Neural-Networks

## Introduction

The Lunar Lander Simulation project aims to automate the landing process of a lunar module on a designated goal using a Multilayer Perceptron (MLP) neural network. The project encompasses the collection of biased data by simulating various goal positions to train the neural network effectively for diverse landing scenarios.

## Technologies

Python 3.x
TensorFlow 2.x
Swift UI (for UI elements)
Matplotlib (for data visualization)

## Features

Real-time simulation of lunar landing scenarios.
Automatic landing mechanism powered by an MLP neural network.
Integrated data collection module for generating training data.
Capability to adapt dynamically to different landing goals.

## Data Collection

Data for the model is collected by simulating different goal positions and potential landing trajectories for the lunar module. This approach enables the model to generalize well across varied landing zones. Biased sampling techniques are employed to ensure that corner cases are sufficiently covered, thereby enhancing the model's robustness.

## Neural Network Architecture

The architecture of the MLP neural network comprises three layers:

Input Layer: Consists of 2 neurons, representing various parameters such as velocity, position, etc.
Hidden Layer: Adjustable numbers of neurons equipped with sigmoid activation functions.
Output Layer: Comprises 2 neurons responsible for controlling thrust and direction.

## Installation

To clone and set up the project on your local machine, execute the following commands:

