from get_data import load_data
from NN import NeuralNetwork
import numpy as np
import time 
import matplotlib.pyplot as plt

def train_model():
    data = load_data()
    NN = NeuralNetwork(input_size= data['x_train'].shape[1], hidden_size=8)

    #Optimized training for CPU
    batch_size=32
    epochs = 500

    #Manual batch training
    n_samples = data['x_train'].shape[0]
    print(f"Training on {n_samples} samples (CPU-based)")

    start_time = time.time()

    for epoch in range(epochs):
        # Mini-batch training
        permuatation = np.random.permutation(n_samples)
        x_shuffled = data['x_train'][permuatation]
        y_shuffled = data['y_train'][permuatation]

        for i in range (0, n_samples, batch_size):
            x_batch = x_shuffled[i: i+batch_size]
            y_batch = y_shuffled[i: i+batch_size]

            NN.feedforward(x_batch)
            NN.backprop(x_batch, y_batch, learning_rate=0.001)

        if epoch % 50 == 0 :
            loss = NN.compute_loss(data['y_train'], NN.feedforward(data['x_train']))
            print(f"EPOCH: {epoch}, LOSS: {loss:.4f} ")

    train_pred = NN.predict(data['x_train'])
    test_pred = NN.predict(data['x_test'])

    train_acc = np.mean(train_pred == data['y_train']) * 100
    test_acc = np.mean(test_pred == data['y_test']) * 100

    print(f"\nTraining completed in {time.time()-start_time:.1f}")
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Test Accuracy {test_acc:.2f}")
def train_and_plot():
    loss_history = []
    train_acc_history = []
    test_acc_history = []

    data = load_data()
    NN = NeuralNetwork(input_size=data['x_train'].shape[1], hidden_size=8)

    batch_size = 32
    epochs = 500
    n_samples = data['x_train'].shape[0]

    for epoch in range(epochs):
        permuatation = np.random.permutation(n_samples)
        x_shuffled = data['x_train'][permuatation]
        y_shuffled = data['y_train'][permuatation]

        for i in range(0, n_samples, batch_size):
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            NN.feedforward(x_batch)
            NN.backprop(x_batch, y_batch, learning_rate=0.001)

        y_train_pred = NN.feedforward(data['x_train'])
        loss = NN.compute_loss(data['y_train'], y_train_pred)
        loss_history.append(loss)

        train_pred = NN.predict(data['x_train'])
        test_pred = NN.predict(data['x_test'])
        train_acc = np.mean(train_pred == data['y_train']) * 100
        test_acc = np.mean(test_pred == data['y_test']) * 100
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

    # Plot it
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, color='crimson')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy', color='blue')
    plt.plot(test_acc_history, label='Test Accuracy', color='green')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    train_model()
    train_and_plot()


