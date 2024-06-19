# README
## FullyConnected Neural Network
This repository contains a Python implementation of a FullyConnected Neural Network from scratch using NumPy. The neural network is designed for digit classification using the MNIST dataset. The network supports ReLU and Softmax activation functions, and it uses the Adam optimizer for training.

## Repository Structure
- main.py: Main script to train and test the neural network.
- Layer.py: Contains the FullyConnectedLayer class, which defines a single layer of the neural network.
- Utils.py: Utility functions for loss calculation, accuracy measurement, and plotting results.
notebook.ipynb: Jupyter notebook for experiments and testing the neural network.

## Usage
Training the Model
To train the neural network, run the main.py script:
python main.py

The script will load the MNIST dataset, define the neural network, train it, and evaluate it on the test set. Training progress will be printed every 10 epochs, and the final test loss and accuracy will be displayed.

## Contributing
Feel free to submit issues, fork the repository, and send pull requests. Contributions are welcome!
