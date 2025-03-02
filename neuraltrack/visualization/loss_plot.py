import json
import os
import matplotlib.pyplot as plt
import argparse

class LossPlotter:
    def __init__(self, log_path="loss_log.json", save_dir="LossPlots"):
        """
        Initializes the LossPlotter.

        Args:
            log_path (str): Path to the loss log JSON file.
            save_dir (str): Directory where plots will be saved.
        """
        self.log_path = log_path
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def load_loss_data(self):
        """Loads the loss data from the JSON file."""
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Loss log file '{self.log_path}' not found.")

        with open(self.log_path, "r") as f:
            self.loss_data = json.load(f)

    def plot_losses(self, show_plot=False):
        """
        Plots the loss curves for each loss component over epochs.

        Args:
            show_plot (bool): Whether to display the plot after saving.
        """
        self.load_loss_data()

        # Extract the epoch numbers and sort them
        epochs = sorted(int(epoch_key.split("_")[-1]) for epoch_key in self.loss_data.keys())
        
        # Identify the loss keys (excluding epoch, time, and total time keys)
        loss_keys = [key for key in self.loss_data[f"epoch_{epochs[0]}"] if key not in ["epoch", "time_for_epoch", "total_time_elapsed"]]

        # Create a plot
        plt.figure(figsize=(10, 6))

        # Plot each loss component over the epochs
        for loss_key in loss_keys:
            loss_values = [self.loss_data[f"epoch_{epoch}"][loss_key] for epoch in epochs]
            plt.plot(epochs, loss_values, marker="o", label=loss_key)

        # Labeling the plot
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Components Over Epochs")
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file
        save_path = os.path.join(self.save_dir, "loss_plot.png")
        plt.savefig(save_path)
        print(f"Loss plot saved at {save_path}")

        # Optionally show the plot interactively
        if show_plot:
            plt.show()

def main():
    """Main function to execute plotting from command line."""
    parser = argparse.ArgumentParser(description="Plot loss curves from a log file.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the loss log file.")
    parser.add_argument("--show_plot", action="store_true", help="Show plot interactively.")
    parser.add_argument("--save_dir", action=str, help="Path where to save the plotting")

    args = parser.parse_args()

    # Initialize the plotter and plot the losses
    plotter = LossPlotter(log_path=args.log_path)
    plotter.plot_losses(show_plot=args.show_plot)

if __name__ == "__main__":
    main()
