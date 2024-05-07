import numpy as np


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation):
        """
        Initialize weights and biases, define the shapes and the
        activation function.

        Args:
            input_size (int): Input shape of the layer
            output_size (int): Output of the layer
            activation (str): activation function
        """
        self.activation = activation

        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

        # define m & v for weights and biases (used in Adam optimization)
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))

        # define hyperparameters (for Adam optimizer)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.

        Args:
            x (Tensor): Input data of shape (batch_size, input_size).
        Returns:
            Tensor: Output of the layer.

        """
        self.x = x
        z = np.dot(self.x, self.weights) + self.biases

        self.activate_activation_function(z)
        return self.output

    def activate_activation_function(self, z: np.ndarray) -> None:
        """
        Apply activation function on the input.

        Args:
            z (Tensor): Input for the activation function.
        """
        if self.activation == "relu":
            self.output = np.maximum(0, z)

        elif self.activation == "softmax":
            exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True))
            sum_values = np.sum(exp_values, axis=-1, keepdims=True)
            self.output = exp_values / sum_values

        else:
            raise ValueError(f"Please define new activation function for the activation you gave")

    def backward(self, d_values: np.ndarray, learning_rate: float, t: int):
        """
        Backpropagation.
        This function will derivative the activation function, and then calculate the
        derivative once with respect to the bias and oe with respect to the weight.
        Those values might be very small, so we will clip them to keep numerical stability.

        Args:
            d_values (float): Derivative of the output
            learning_rate (float): Learning rate for gradient descent
            t (int): Timestep
        Returns:
            Tensor: Derivative with respect to the inputs.
        """
        d_values = self.derivative_activation_function(d_values)

        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)

        d_biases, d_weights = self.clipping(d_biases, d_weights)

        # Calculate the gradient with respect to the input
        d_inputs = np.dot(d_values, self.weights.T)

        # Update the weights and biases using the learning rate and their derivatives
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        # Update weights & biases using m and v values
        m_weights, v_weights = self.update_parameters({'param': self.weights, 'm': self.m_weights, 'v': self.v_weights},
                                                      d_weights, t, learning_rate)
        m_biases, v_biases = self.update_parameters({'param': self.biases, 'm': self.m_biases, 'v': self.v_biases},
                                                    d_biases, t, learning_rate)
        return d_inputs

    def update_parameters(self, parameters, d_parameters, t, learning_rate):
        """
        Update parameters using Adam optimizer.

        Args:
            parameters (dict): Dictionary containing parameters and their moment estimates.
            d_parameters (Tensor): Gradient of parameters.
            t (int): Timestep.
            learning_rate (float): Learning rate.

        Returns:
            Tuple[np.Tensor, np.Tensor]: Updated moment estimates.
        """
        m = self.beta1 * parameters['m'] + (1 - self.beta1) * d_parameters
        v = self.beta2 * parameters['v'] + (1 - self.beta2) * (d_parameters ** 2)
        m_hat = m / (1 - self.beta1 ** t)
        v_hat = v / (1 - self.beta2 ** t)

        # Update parameters
        parameters['param'] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return m, v

    def clipping(self, d_biases, d_weights):
        """
        Clip gradients to avoid exploding gradients.

        Args:
            d_biases (Tensor): Gradient of biases.
            d_weights (Tensor): Gradient of weights.

        Returns:
            Tuple[Tensor, Tensor]: Clipped gradients.
        """
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)
        return d_biases, d_weights

    def derivative_activation_function(self, d_values: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the activation function.

        Args:
            d_values (Tensor): Derivative of the output.

        Returns:
            Tensor: Derivative of the activation function.
        """
        if self.activation == "softmax":
            for i, gradient in enumerate(d_values):
                if len(gradient.shape) == 1:  # single instance case
                    gradient = gradient.reshape(-1, 1)
                jacobian_matrix = np.diagflat(gradient) - np.dot(gradient, gradient.T)
                d_values[i] = np.dot(jacobian_matrix, self.output[i])

        # Calculate the derivative of the ReLU function
        elif self.activation == "relu":
            d_values = d_values * (self.output > 0)
        return d_values

