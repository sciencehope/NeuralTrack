import json
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re

class GradientPlotter:
    def __init__(self, log_path="gradient_log.json", save_dir="GradientPlots", chart_type="bar", include_bias=True):
        """
        Initializes the GradientPlotter.

        Args:
            log_path (str): Path to the gradient log JSON file.
            save_dir (str): Directory where plots will be saved.
            chart_type (str): Type of chart to generate ('bar' or 'line').
            include_bias (bool): Whether to include bias layers in the plot.
        """
        self.log_path = log_path
        self.save_dir = save_dir
        self.chart_type = chart_type
        self.include_bias = include_bias

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_gradient_data(self):
        """Loads the gradient data from the JSON file."""
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Gradient log file '{self.log_path}' not found.")

        with open(self.log_path, "r") as f:
            self.gradient_data = json.load(f)

    def plot_gradients(self, show_plot=False, epoch=None):
        """
        Plots the gradient statistics (mean, median, max, min) over epochs.

        Args:
            show_plot (bool): Whether to display the plot interactively.
            epoch (int, optional): The specific epoch to plot. If None, all epochs will be plotted.
        """
        self.load_gradient_data()

        # Sort epochs
        all_epochs = sorted(int(epoch_key.split("_")[-1]) for epoch_key in self.gradient_data.keys())

        # If a specific epoch is provided, check if it exists
        if epoch is not None:
            if epoch not in all_epochs:
                raise ValueError(f"Epoch {epoch} not found in the gradient log file.")
            epochs_to_plot = [epoch]
        else:
            epochs_to_plot = all_epochs  # Plot all epochs if none specified

        for epoch in epochs_to_plot:
            epoch_key = f"epoch_{epoch}"
            layers = []
            means = []
            medians = []
            max_vals = []
            min_vals = []

            for layer_name, stats in self.gradient_data[epoch_key].items():
                if 'epoch' in layer_name or 'time' in layer_name:
                    continue  # Skip epoch and time keys

                # Handle bias layers with various naming patterns
                if re.search(r'\.bias|bias|\(bias\)', layer_name):
                    if not self.include_bias:
                        continue  # Skip bias layers if not needed
                    clean_layer_name = f"bias{re.sub(r'\D', '', layer_name)}"
                else:
                    clean_layer_name = layer_name.replace(".weight", "")

                layers.append(clean_layer_name)
                means.append(stats["mean"])
                medians.append(stats["median"])
                max_vals.append(stats["max"])
                min_vals.append(stats["min"])

            indices = np.arange(len(layers))
            plt.figure(figsize=(10, 6), constrained_layout=True)

            if self.chart_type == 'bar':
                plt.bar(indices - 0.3, means, width=0.2, label="Mean", color="blue")
                plt.bar(indices - 0.1, medians, width=0.2, label="Median", color="green")
                plt.bar(indices + 0.1, max_vals, width=0.2, label="Max", color="red")
                plt.bar(indices + 0.3, min_vals, width=0.2, label="Min", color="purple")
            elif self.chart_type == 'line':
                plt.plot(indices - 0.3, means, marker="o", label="Mean", color="blue", linestyle='-', linewidth=2)
                plt.plot(indices - 0.1, medians, marker="o", label="Median", color="green", linestyle='-', linewidth=2)
                plt.plot(indices + 0.1, max_vals, marker="o", label="Max", color="red", linestyle='-', linewidth=2)
                plt.plot(indices + 0.3, min_vals, marker="o", label="Min", color="purple", linestyle="-", linewidth=2)

            plt.xticks(indices, layers, rotation=45, ha="right")
            plt.xlabel("Layers")
            plt.ylabel("Gradient Value")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)

            save_path = os.path.join(self.save_dir, f"gradient_plot_epoch_{epoch}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"Gradient plot for epoch {epoch} saved at {save_path}")

            if show_plot:
                plt.show()

            plt.close()




def main():
    parser = argparse.ArgumentParser(description="Plot gradient norms from a log file.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the gradient log file.")
    parser.add_argument("--save_dir", type=str, required=False, help="Path to the gradient log file.", default="GradientPlots")
    parser.add_argument("--chart_type", type=str, choices=['bar', 'line'], required=False, help="Type of the Chart", default="line")
    parser.add_argument("--show_plot", action="store_true", help="Show plot interactively.")
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--include_bias', type=bool, default=False)

    args = parser.parse_args()

    plotter = GradientPlotter(log_path=args.log_path, save_dir=args.save_dir, chart_type=args.chart_type, include_bias=args.include_bias)
    plotter.plot_gradients(show_plot=args.show_plot, epoch=args.epoch)

if __name__ == "__main__":
    from gradient_plot import GradientPlotter  # Local import inside the script
    main()