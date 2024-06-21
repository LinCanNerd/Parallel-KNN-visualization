import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Parameters
n_samples = 100000 # Total number of points
n_features = 2    # Number of features (x, y)
n_classes = 5     # Number of classes
x_range = (0, 720)  # Range for x coordinates
y_range = (0 ,720)  # Range for y coordinates
spread = 50 # Spread of the data points

# Generate synthetic data
centers = [(np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])) for _ in range(n_classes)]
X, y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=spread, n_features=n_features, random_state=42)

# Create a DataFrame
df = pd.DataFrame(data=X, columns=['x', 'y']).astype(int).abs()
df['class'] = y
# Save to CSV
df.to_csv('dataset/centomila.csv', index=False, header=False)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(df['x'], df['y'], c=df['class'], cmap='viridis', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Data for KNN Training')
plt.colorbar(label='Class')
plt.xlim(x_range)
plt.ylim(y_range)
plt.show()

print("Dataset created and saved as 'knn_dataset.csv'")
