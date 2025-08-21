import numpy as np
from tensorflow.keras.datasets import mnist

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(x.dtype)

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred))

def train_model():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    n_classes = 10

    # Reshape and normalize data
    X_train = X_train.reshape(60000, 784).T / 255.0  # Transpose for matrix multiplication
    y_train = np.eye(n_classes)[y_train].T

    X_test = X_test.reshape(10000, 784).T / 255.0
    y_test = y_test

    # He Initialization
    layer_weights_A = np.random.randn(256, 784) * np.sqrt(2. / 784)
    layer_bias_A    = np.zeros((256, 1))

    layer_weights_B = np.random.randn(10, 256) * np.sqrt(2. / 256)
    layer_bias_B    = np.zeros((10, 1))

    # Training loop
    lr = 0.01
    batch_size = 100
    epochs = 60

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_train[:, i:i+batch_size]
            y_batch = y_train[:, i:i+batch_size]

            # Forward pass
            Z1 = np.dot(layer_weights_A, X_batch) + layer_bias_A
            A1 = relu(Z1)  # 👈 Activation function added here!
            Z2 = np.dot(layer_weights_B, A1) + layer_bias_B
            A2 = softmax(Z2)

            # Calculate loss for the batch
            loss = categorical_cross_entropy(y_batch, A2) / batch_size
            total_loss += loss

            # Backpropagation
            dZ2 = A2 - y_batch
            dW2 = np.dot(dZ2, A1.T) / batch_size
            db2 = np.sum(dZ2, axis=1, keepdims=True) / batch_size

            dZ1 = np.dot(layer_weights_B.T, dZ2) * relu_derivative(Z1) # 👈 Derivative of ReLU
            dW1 = np.dot(dZ1, X_batch.T) / batch_size
            db1 = np.sum(dZ1, axis=1, keepdims=True) / batch_size

            # Update weights
            layer_weights_B -= lr * dW2
            layer_bias_B    -= lr * db2
            layer_weights_A -= lr * dW1
            layer_bias_A    -= lr * db1
            

        avg_loss = total_loss / (X_train.shape[1] / batch_size)
        print(f'Epoch {epoch + 1}: Loss = {avg_loss:.4f}')

    print(f'FINISHED TRAINING')

    # Evaluation
    Z1_test = np.dot(layer_weights_A, X_test) + layer_bias_A
    A1_test = relu(Z1_test)
    Z2_test = np.dot(layer_weights_B, A1_test) + layer_bias_B
    A2_test = softmax(Z2_test)

    predictions = np.argmax(A2_test, axis=0)
    correct_predictions = np.sum(predictions == y_test)
    accuracy = correct_predictions / y_test.shape[0]

    print(f'Accuracy: {accuracy:.4f}')


print('Start model\n----------------------------------')
train_model()
