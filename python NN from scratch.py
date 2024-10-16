import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# set the seed to have same random numbers at each trial
np.random.seed(42)

# Initialize neural network parameters
def initialize_parameters(input_size, hidden_size, output_size):
    # Randomly initialize weights and biases
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.random.randn(hidden_size)
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.random.randn(output_size)
    return W1, b1, W2, b2

# Feedforward pass
def feedforward(X, W1, b1, W2, b2):
    # Input to hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    
    # Hidden to output layer
    Z2 = np.dot(A1, W2) + b2 
    A2 = sigmoid(Z2)
    
    return A1, A2

# Backpropagation
def backpropagation(X, Y, A1, A2, W2, learning_rate):
    # Calculate error at output layer
    output_error = A2 - Y
    output_delta = output_error * sigmoid_derivative(A2)
    
    # Calculate error at hidden layer
    hidden_error = np.dot(output_delta, W2.T)
    hidden_delta = hidden_error * sigmoid_derivative(A1)
    
    # Return the gradients
    return hidden_delta, output_delta  # W(new) = W(old) - (learning_rate * gradients)


# Update weights and biases
def update_parameters(W1, b1, W2, b2, A1, X, hidden_delta, output_delta, learning_rate):
    W2 -= learning_rate * np.dot(A1.T, output_delta)
    b2 -= learning_rate * np.sum(output_delta, axis=0)
    
    W1 -= learning_rate * np.dot(X.T, hidden_delta)
    b1 -= learning_rate * np.sum(hidden_delta, axis=0)
     
    return W1, b1, W2, b2

# Loss function (Mean Squared Error)
def mean_squared_error(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

# Neural network training
def train_neural_network(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    # Initialize parameters
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        # Feedforward
        A1, A2 = feedforward(X, W1, b1, W2, b2)
        
        # Calculate the loss
        loss = mean_squared_error(Y, A2)
        
        # Backpropagation
        hidden_delta, output_delta = backpropagation(X, Y, A1, A2, W2, learning_rate)
        
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, A1, X, hidden_delta, output_delta, learning_rate)
        
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, b1, W2, b2

# Example usage
if __name__ == "__main__":
    # sample input
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],[2,4]])
    Y = np.array([[0], [1], [1], [0],[1]])
    
    input_size = 2  # Number of input features
    hidden_size = 3  # Number of neurons in hidden layer
    output_size = 1  # Number of output neurons
    
    # Train the neural network
    trained_W1, trained_b1, trained_W2, trained_b2 = train_neural_network(
        X, Y, input_size, hidden_size, output_size, epochs=10000, learning_rate=0.1
    )

    # Test the network
    _, predictions = feedforward(X, trained_W1, trained_b1, trained_W2, trained_b2)
    
    print("Predictions:")
    print(predictions)

# Apply threshold to convert continuous predictions to binary
binary_predictions = (predictions >= 0.5).astype(int)

print("Binary Predictions (0 or 1):")
print(binary_predictions)

# example input 

import numpy as np

input_str = input("Enter numbers separated by space: ")  # Get input as a string
input_list = list(map(int, input_str.split()))  # Split the string and convert to integers
input_1 = np.array(input_list)  # Create a NumPy array


_, predictions_1 = feedforward(input_1, trained_W1, trained_b1, trained_W2, trained_b2)
binary = (predictions_1 >= 0.5).astype(int)
    
print("Predictions:") 
print(binary)
