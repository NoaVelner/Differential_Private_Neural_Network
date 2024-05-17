import numpy as np
from typing import List
import matplotlib.pyplot as plt


def accuracy(output: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculates the accuracy of the model predictions.

    Args:
        output (Tensor): Model predictions of shape (num_samples, output_size).
        targets (Tensor): Target labels of shape (num_samples, output_size).

    Returns:
        float: Accuracy value.
    """
    predicted_labels = np.argmax(output, axis=1)
    true_labels = np.argmax(targets, axis=1)
    return np.mean(predicted_labels == true_labels)


def plot_training(accuracy_log: List[float], loss_log: List[float], n_epochs: int) -> None:
    """
    Plots the training curves.

    Args:
        accuracy_log (List[float]): List of accuracies.
        loss_log (List[float]): List of losses.
        n_epochs (int): Number of training epochs.
    """
    plt.plot(range(n_epochs), loss_log, label='Training Loss')
    plt.plot(range(n_epochs), accuracy_log, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training Loss and Accuracy')
    plt.legend()
    plt.show()


def calculate_loss_crossentropy(output, targets):
    """
    Calculates the categorical cross-entropy loss.

    Args:
        output (Tensor): Model predictions of shape (num_samples, output_size).
        targets (Tensor): Target labels of shape (num_samples, output_size).

    Returns:
        float: Categorical cross-entropy loss value.
    """
    epsilon = 1e-10
    loss = -np.mean(targets * np.log(output + epsilon))
    return loss
