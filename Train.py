from typing import List
import numpy as np
import Utils
from Layer import FullyConnectedLayer  # Import custom layer

# load data & preprocessing
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


class CreateModel:
    def __init__(self, input_size: int, output_size: int, hidden_size: List[int]):
        """
        Creates a feedforward neural network model.

        Args:
            :arg input_size (int): Number of input features.
            :arg output_size (int): Number of output classes.
            :arg hidden_size (List[int]): List of hidden layer sizes.
        """
        self.layer1 = FullyConnectedLayer(input_size=input_size, output_size=hidden_size[0], activation="relu")
        self.layer2 = FullyConnectedLayer(input_size=hidden_size[0], output_size=hidden_size[1], activation="relu")
        self.layer3 = FullyConnectedLayer(input_size=hidden_size[1], output_size=output_size, activation="softmax")

    def forward(self, inputs):
        """
        Performs forward propagation through the network.

        Args:
            inputs (Tensor): Input data of shape (batch_size, input_size).
        Returns:
            (Tensor): Output predictions of shape (batch_size, output_size).
        """
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)
        return output3

    def train(self, inputs, targets, n_epochs, initial_learning_rate, decay, plot_training_results=False):
        """
        Trains the neural network model.
**
        This function does the training process of the model,
            First forward propagation is done, then the loss and accuracy are calculated,
            After that the backpropagation is done.
**
        Args:
            inputs (Tensor): Input training data of shape (num_samples, input_size).
            targets (Tensor): Target labels of shape (num_samples, output_size).
            n_epochs (int): Number of training epochs.
            initial_learning_rate (float): Initial learning rate.
            decay (float): Learning rate decay factor.
            plot_training_results (bool, optional): Whether to plot training results. Defaults = False.
        """
        # Initialize timestamp and accuracy & loss.
        time = 0
        loss_log = []
        accuracy_log = []

        for epoch in range(n_epochs):
            output = self.forward(inputs=inputs)

            loss = Utils.calculate_loss_crossentropy(output, targets)
            accuracy = Utils.accuracy(output, targets)
            self.backward(decay, epoch, initial_learning_rate, output, targets, time)

            if plot_training_results:
                loss_log.append(loss)
                accuracy_log.append(accuracy)
            print(f"Epoch {epoch} \nLoss: {loss} \nAccuracy: {accuracy}")

        if plot_training_results:
            Utils.plot_training(accuracy_log, loss_log, n_epochs)

    def backward(self, decay, epoch, initial_learning_rate, output, targets, time):
        output_grad = 6 * (output - targets) / output.shape[0]
        time += 1
        learning_rate = initial_learning_rate / (1 + decay * epoch)
        grad_3 = self.layer3.backward(output_grad, learning_rate, time)
        grad_2 = self.layer2.backward(grad_3, learning_rate, time)
        grad_1 = self.layer1.backward(grad_2, learning_rate, time)


def load_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Flatten the images
    X_train = X_train.reshape((60000, 784))
    X_train = X_train.astype("float32") / 255.0
    Y_train = to_categorical(Y_train)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    input_shape = 784
    hidden_shape = [512, 512]
    output_shape = 10

    nn = CreateModel(input_size=input_shape, output_size=output_shape, hidden_size=hidden_shape)
    nn.train(x_train, y_train, initial_learning_rate=0.001, decay=0.001, n_epochs=200, plot_training_results=True)
