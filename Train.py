from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

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

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation through the network.

        Args:
            :param inputs: (np.ndarray) Input data of shape (batch_size, input_size).
        Returns:
            Output predictions of shape (batch_size, output_size).
        """
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)
        return output3

    def train(self, inputs: np.ndarray, targets: np.ndarray, n_epochs: int, initial_learning_rate: float,
              decay: float, plot_training_results: bool = False) -> None:
        # def train(self, inputs, targets, n_epochs, initial_learning_rate, decay, plot_training_results=False):
        """
        Trains the neural network model.
        This function does the training process of the model,
            First forward propagation is done, then the loss and accuracy are calculated,
            After that the backpropagation is done.

        Args:
            :param inputs: (Tensor) Input training data of shape (num_samples, input_size).
            :param targets: (Tensor) Target labels of shape (num_samples, output_size).
            :param n_epochs: (int) Number of training epochs.
            :param initial_learning_rate: (float) Initial learning rate.
            :param decay:  (float) Learning rate decay factor.
            :param plot_training_results: (bool, optional) Whether to plot training results. Defaults = False.
        """
        time = 0
        loss_log = []
        accuracy_log = []

        for epoch in range(n_epochs):
            output = self.forward(inputs=inputs)
            loss = Utils.calculate_loss_crossentropy(output, targets)
            accuracy = Utils.accuracy(output, targets)

            self.backward(decay, epoch, initial_learning_rate, output, targets, time)
            time += 1  # modified 17.5

            if plot_training_results:
                loss_log.append(loss)
                accuracy_log.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Loss: {loss} - Accuracy: {accuracy}")

        if plot_training_results:
            Utils.plot_training(accuracy_log, loss_log, n_epochs)

    def backward(self, decay: float, epoch: int, initial_learning_rate: float, output: np.ndarray,
                 targets: np.ndarray, time: int) -> None:

        # def backward(self, decay, epoch, initial_learning_rate, output, targets, time):
        """
        Performs backward propagation through the network.

        Args:
            decay: Learning rate decay factor.
            epoch: Current epoch number.
            initial_learning_rate: Initial learning rate.
            output: Output predictions of shape (batch_size, output_size).
            targets: Target labels of shape (batch_size, output_size).
            time: Current time step.
        """
        output_grad = 6 * (output - targets) / output.shape[0]
        time += 1
        learning_rate = initial_learning_rate / (1 + decay * epoch)
        grad_3 = self.layer3.backward(output_grad, learning_rate, time, noise_factor=0.01/(1 + decay * epoch))
        grad_2 = self.layer2.backward(grad_3, learning_rate, time, noise_factor=0.01/(1 + decay * epoch))
        grad_1 = self.layer1.backward(grad_2, learning_rate, time, noise_factor=0.01/(1 + decay * epoch))

    def get_prediction(self, samples):
        return self.forward(inputs=samples)

    def test_loss(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the test loss of the model.

        Args:
            model (CreateModel): The trained neural network model.
            x_test (np.ndarray): Test input data of shape (num_samples, input_size).
            y_test (np.ndarray): Test target labels of shape (num_samples, output_size).

        Returns:
            float: Test loss.
        """
        predictions = self.get_prediction(x_test)
        return Utils.calculate_loss_crossentropy(predictions, y_test)

    def test_accuracy(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Calculate the test accuracy of the model.

        Args:
            model (CreateModel): The trained neural network model.
            x_test (np.ndarray): Test input data of shape (num_samples, input_size).
            y_test (np.ndarray): Test target labels of shape (num_samples, output_size).

        Returns:
            float: Test accuracy.
        """
        # predictions = model.forward(inputs=x_test)
        predictions = self.get_prediction(x_test)
        predictions = np.argmax(predictions, axis=1)
        y_test = np.argmax(y_test, axis=1)
        return np.mean(predictions == y_test)


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing train and test data.
    """
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((60000, 784))
    X_train = X_train.astype("float32") / 255.0
    Y_train = to_categorical(Y_train)
    return X_train, Y_train, X_test, Y_test



if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist()

    input_shape = 784
    hidden_shape = [512, 512]
    output_shape = 10
    x_test = x_test.reshape((x_test.shape[0], -1))
    y_test = to_categorical(y_test, num_classes=output_shape)

    nn = CreateModel(input_size=input_shape, output_size=output_shape, hidden_size=hidden_shape)
    nn.train(x_train, y_train, initial_learning_rate=0.001, decay=0.001, n_epochs=100, plot_training_results=True)
    print("Test Loss:", nn.test_loss(x_test, y_test))
    print("Test Accuracy:", nn.test_accuracy(x_test, y_test))