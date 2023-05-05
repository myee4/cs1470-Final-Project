'''
Partially sourced from ChatGPT (GPT-4) to help plot distributions of the data that
we sourced from Kaggle. This was to make sure that the distribution of data was reasonable.
'''

import matplotlib.pyplot as plt
import pandas as pd

view_data = pd.read_csv("./kaggle_data/one_data_to_rule_them_all.csv")

# Sample series of numbers
data = view_data['view_count']

def plot_distribution(data, bins=1000):
    # Create a histogram using matplotlib
    plt.hist(data, bins=bins, edgecolor='black')

    # Set the title and labels
    plt.title("Data Distribution")
    plt.xlabel("Values")
    plt.ylabel("Frequency")

    # Show the plot
    plt.show()

# Call the function with the data
plot_distribution(data)
