import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(file):

    df = pd.read_csv('data/AI_vehicle_theft_data.csv', header=0)

    # Remove bad data, chatgpt is not perfect
    df = df[df['horsepower'] != 0]

    X = df[['year', 'production', 'horsepower', 'reliability', 'price']]
    y = df['rate']

    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # Add bias term
    return X_scaled

def gradient_descent(X_train, y_train, alpha=0.01, iterations=1000):
    m = len(y_train)
    theta = np.zeros(X_train.shape[1])  
    
    cost_history = []
    
    for _ in range(iterations):
        # Hypothesis
        y_pred = np.dot(X_train, theta)
        
        # Cost function
        cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
        cost_history.append(cost)
        
        # Gradients
        gradients = (1 / m) * np.dot(X_train.T, (y_pred - y_train))
        
        # Update parameters
        theta -= alpha * gradients
        
    return theta, cost_history

def evaluate_model(X_test, y_test, theta):
    y_pred_test = np.dot(X_test, theta)
    mse = mean_squared_error(y_test, y_pred_test)
    return mse, y_pred_test

def plot_cost_function(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost function convergence')
    # plt.savefig('code/figures/CostFunctionConverge.pdf')

    plt.show()

def plot_actual_vs_predicted(y_test, y_pred_test):
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual Theft Rate')
    plt.ylabel('Predicted Theft Rate')
    plt.title('Actual vs Predicted Theft Rate')
    # plt.savefig('code/figures/ActualvsPredicted.pdf')
    plt.show()

def main():
    
    # Load data
    X, y = load_data('data/AI_vehicle_theft_data.csv')

    # Scale data
    X = scale_features(X)

    # Train test split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the model using gradient descent
    theta, cost_history = gradient_descent(X_train, y_train, alpha=0.01, iterations=1000)

    # Evaluate the model on the test set
    mse, y_pred_test = evaluate_model(X_test, y_test, theta)
    print(f"Mean Squared Error on Test Set: {mse}")

    # Plot actual vs predicted values
    plot_actual_vs_predicted(y_test, y_pred_test)

    # Plot cost function to check convergence
    plot_cost_function(cost_history)

    # Output the learned parameters (theta values)
    print("Learned Parameters (Theta):", theta)


if __name__ == '__main__':
    main()