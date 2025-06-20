import seaborn as sns
import matplotlib.pyplot as plt

def map_correlations(correlation_matrix):
    plt.figure(figsize=(round(len(correlation_matrix) / 4), round(len(correlation_matrix) / 4)))
    sns.heatmap(correlation_matrix, annot=False, cmap='RdYlBu_r', center=0, square=True, linewidths=0.1)
    plt.title('Stock Correlation Matrix')
    plt.show()