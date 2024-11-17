
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        return self.sigmoid(np.dot(self.hidden, self.weights_hidden_output) + self.bias_output)

    def backpropagate(self, X, Y, lr):
        output = self.feedforward(X)
        output_error = Y - output
        output_delta = output_error * self.sigmoid_derivative(output)
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)
        
        self.weights_hidden_output += self.hidden.T.dot(output_delta) * lr
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * lr
        self.weights_input_hidden += X.T.dot(hidden_delta) * lr
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * lr

    def train(self, X, Y, epochs, lr):
        for epoch in range(epochs):
            self.backpropagate(X, Y, lr)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(Y - self.feedforward(X)))
                print(f"Epoch {epoch}, Loss: {loss}")

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
    nn.train(X, Y, epochs=10000, lr=0.1)

    print("\nPredictions after training:")
    print(nn.feedforward(X))


