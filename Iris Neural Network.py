from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Iris.csv
iris = pd.read_csv('D:/Windows D/Downloads/irisdata (1).csv')

# Reading and seperating data
X = iris.iloc[:, :-1].values  # Features 
y = iris.iloc[:, -1].values  # Target iris

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = OneHotEncoder(sparse_output=False).fit_transform(y_encoded.reshape(-1, 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) # 80% of the data would be used for training and 20% for testing

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for dimensionality reduction for visualization purposes
pca = PCA(n_components=2)  # Reduce to 2D for plotting
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Define the neural network
model = Sequential([
    Input(shape=(2,)),  # Input shape is 2D
    Dense(16, activation='relu'),  
    Dense(12, activation='relu'),
    Dense(3, activation='softmax')  # 3 output classes for Iris dataset: Setosa, Versicolor, Virginica
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pca, y_train, epochs=50, batch_size=8, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pca, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Function to plot decision boundary
def plot_decision_boundary(X, y, model, resolution=0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), np.arange(y_min, y_max, resolution))
    
    # Predict on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid_points)
    predictions = np.argmax(predictions, axis=1).reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, predictions, alpha=0.5, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title("Decision Boundary")
    plt.xlabel("PCA Component 1") # Linear combination of all of the iris features: sepal length, width and pedal length, wdith
    plt.ylabel("PCA Component 2") # Direction of maximum variance in the data that is orthogonal to the first principal component
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_train_pca, y_train, model)

# Learning Rate Analysis
learning_rates = [0.001, 0.01, 0.1] # Different learning rates to be used for comparisons purposes
histories = [] # Performance of the learning rates within an array

for lr in learning_rates:
    model = Sequential([
        Input(shape=(4,)),  # Explicit Input layer to avoid warning
        Dense(8, activation='relu'), # Using Relu as it is most commonly used now
        Dense(6, activation='relu'), # Using Relu as it is most commonly used now
        Dense(3, activation='softmax') 
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)
    histories.append(history.history['loss'])

for i, lr in enumerate(learning_rates):
    plt.plot(histories[i], label=f'LR={lr}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Rate Comparison')
plt.show()

# Testing different non-linearities
activation_functions = ['relu', 'tanh', 'sigmoid'] # Testing different nonlinearities
activation_lr_histories = {}

for activation in activation_functions:
    for lr in learning_rates: # using the same learning rates
        print(f"Training and Testing with activation function: {activation} and learning rate: {lr}")
        
        # Define the neural network
        model = Sequential([
            Input(shape=(2,)),  # PCA reduced input
            Dense(16),
            Activation(activation),  # Nonlinearity 1
            Dense(12),
            Activation(activation),  # Nonlinearity 2
            Dense(3, activation='softmax')  # Output layer (softmax for classification)
        ])
        
        # Compile the model with the specified learning rate
        model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(X_train_pca, y_train, validation_data=(X_test_pca, y_test), epochs=50, batch_size=8, verbose=0)
        
        # Store the history for comparison
        activation_lr_histories[(activation, lr)] = history.history
        
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test_pca, y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot training losses for different activation functions and learning rates
plt.figure(figsize=(12, 8))
for (activation, lr), history in activation_lr_histories.items():
    plt.plot(history['loss'], label=f'{activation} - lr={lr}')
    
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Activation Function and Learning Rate Comparison (Training Loss)')
plt.show()
