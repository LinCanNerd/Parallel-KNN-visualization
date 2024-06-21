import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x):
        # Compute the distances between x and all examples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Example usage:
if __name__ == "__main__":
    # Load data from CSV file without headers
    csv_file = 'data.csv'  # Path to your CSV file
    data = pd.read_csv(csv_file, header=None)

    # Separate features and labels
    X_train = data.iloc[:, [0, 1]].values
    y_train = data.iloc[:, 2].values

    # Point to predict
    x_test = np.array([10,10])  # Replace with your test point

    # Initialize the model, fit it on the training data, and make a prediction
    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    prediction = knn.predict(x_test)
    
    # Print the prediction
    print("Prediction for point", x_test, ":", prediction)
