import csv
import matplotlib.pyplot as plt

def read_dataset(filename):
    points = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x, y, label = float(row[0]), float(row[1]), int(row[2])
            points.append((x, y, label))
    return points

def plot_points(points):
    # Create lists for each class
    classes = {i: ([], []) for i in range(5)}
    
    for p in points:
        x, y, label = p
        classes[label][0].append(x)
        classes[label][1].append(y)
    
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    for i in range(5):
        plt.scatter(classes[i][0], classes[i][1], color=colors[i], label=labels[i])
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Generated Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    filename = "data.csv"
    points = read_dataset(filename)
    plot_points(points)
