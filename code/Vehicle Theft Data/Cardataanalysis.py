import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load and Preprocess the Data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Encoding categorical columns
    df['Fuel_Type'] = df['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
    df['Seller_Type'] = df['Seller_Type'].map({'Dealer': 0, 'Individual': 1})
    df['Transmission'] = df['Transmission'].map({'Manual': 0, 'Automatic': 1})
    
    # Fill missing values with median for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns  # Get numeric columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Select features and target variable
    X = df[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission']]
    y = df['Selling_Price']
    
    return X, y
# 2. Feature Scaling (Standardization)
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # Add bias term
    return X_scaled


# 3. Gradient Descent for Linear Regression
def gradient_descent(X_train, y_train, alpha=0.01, iterations=1000):
    m = len(y_train)
    theta = np.zeros(X_train.shape[1])  # Initialize parameters
    
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

# 4. Plot Cost Function to Visualize Convergence
def plot_cost_function(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost function convergence')
    plt.show()

# 5. Model Evaluation: Predictions and Mean Squared Error
def evaluate_model(X_test, y_test, theta):
    y_pred_test = np.dot(X_test, theta)
    mse = mean_squared_error(y_test, y_pred_test)
    return mse, y_pred_test

# 6. Plot Actual vs Predicted Values
def plot_actual_vs_predicted(y_test, y_pred_test):
    plt.scatter(y_test, y_pred_test)
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.show()

# Main function to run the entire process
def main(file_path):
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)
    
    # Scale features
    X_scaled = scale_features(X)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train the model using gradient descent
    theta, cost_history = gradient_descent(X_train, y_train, alpha=0.01, iterations=1000)
    
    # Plot cost function to check convergence
    plot_cost_function(cost_history)
    
    # Evaluate the model on the test set
    mse, y_pred_test = evaluate_model(X_test, y_test, theta)
    print(f"Mean Squared Error on Test Set: {mse}")
    
    # Plot actual vs predicted values
    plot_actual_vs_predicted(y_test, y_pred_test)
    
    # Output the learned parameters (theta values)
    print("Learned Parameters (Theta):", theta)


if __name__ == "__main__":
    file_path = 'data/car data.csv'  
    main(file_path)