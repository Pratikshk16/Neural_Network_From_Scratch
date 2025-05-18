import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data

# Neural Net layer and activation classes
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Init NNFS
nnfs.init()

# Streamlit UI
st.title("🧠 Neural Network Playground (from Scratch)")
st.sidebar.title("Input Coordinates")
x1 = st.sidebar.slider("X1", -1.0, 1.0, 0.0, 0.01)
x2 = st.sidebar.slider("X2", -1.0, 1.0, 0.0, 0.01)

# Data
X_user = np.array([[x1, x2]])

# Network layers
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Forward pass
dense1.forward(X_user)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Prediction
pred = np.argmax(activation2.output[0])
st.write(f"### Predicted class: {pred}")
st.write(f"Probabilities: {activation2.output[0]}")

# Plotting
def plot_decision_boundary():
    plt.clf()
    h = 0.01
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]

    dense1.forward(grid)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    Z = np.argmax(activation2.output, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.6)
    plt.scatter(X_user[:, 0], X_user[:, 1], c='black', edgecolors='k', s=100)
    st.pyplot(plt.gcf())

plot_decision_boundary()
