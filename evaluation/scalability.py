# evaluation/scalability.py

import matplotlib.pyplot as plt
import numpy as np

def plot_scalability(ris_sizes, semantic_accuracies, title='RIS Scalability Study', save_path=None):
    """
    Plot semantic accuracy vs RIS element size.

    Args:
        ris_sizes (list): List of RIS element counts.
        semantic_accuracies (list): Corresponding semantic accuracy results.
        title (str): Plot title.
        save_path (str): Where to save the plot.
    """
    plt.figure()
    plt.plot(ris_sizes, semantic_accuracies, marker='s')
    plt.xlabel('Number of RIS Elements')
    plt.ylabel('Semantic Accuracy')
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()
