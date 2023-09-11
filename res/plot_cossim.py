import json
import matplotlib.pyplot as plt
import argparse
import os
import glob
import re

def plot_cossim_sections(json_file):
    # Load the JSON data
    with open(json_file, 'r') as json_file:
        data = json.load(json_file)

    # Extract the cossim data
    cossim_layer_train = data['estimator']['cossim_layer_train']
    cossim_layer_test = data['estimator']['cossim_layer_test']
    cossim_amm_train = data['estimator']['cossim_amm_train']
    cossim_amm_test = data['estimator']['cossim_amm_test']

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot cossim_layer_train
    axs[0, 0].bar(range(len(cossim_layer_train)), cossim_layer_train)
    axs[0, 0].set_title('Layer CosSim (Train)')

    # Plot cossim_layer_test
    axs[0, 1].bar(range(len(cossim_layer_test)), cossim_layer_test)
    axs[0, 1].set_title('Layer CosSim (Test)')

    # Plot cossim_amm_train
    axs[1, 0].bar(range(len(cossim_amm_train)), cossim_amm_train)
    axs[1, 0].set_title('AMM CosSim (Train)')

    # Plot cossim_amm_test
    axs[1, 1].bar(range(len(cossim_amm_test)), cossim_amm_test)
    axs[1, 1].set_title('AMM CosSim (Test)')

    # Add a common title
    plt.suptitle(f'{data["model"]["name"]} - {data["estimator"]["method"]}', fontsize=16)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot as a PNG file
    output_file = str(json_file)+'.plt.png'
    plt.savefig(output_file)

def main():
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(description='Plot and save CosSim data from JSON files.')

    # Add an argument for the regex pattern
    parser.add_argument('pattern', type=str, help='Regex pattern to match JSON files')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get a list of matching JSON files in the current directory
    files = glob.glob(args.pattern)

    # Iterate through the matching files and plot/save them
    for json_file in files:
        if os.path.isfile(json_file):
            plot_cossim_sections(json_file)
            print(f'Plotted and saved {json_file} as {json_file.replace(".json", ".plt.png")}')

if __name__ == "__main__":
    main()

