import json
import os
import torch
import time
import numpy as np

class GradientLogger:
    def __init__(self, log_path="gradient_log.json", save_every=10, log_interval=10):
        """
        Initializes the gradient logger.

        Args:
            log_path (str): Path to save gradient logs.
            log_interval (int): How often to save gradients (e.g., every 10 epochs).
        """
        self.log_path = log_path
        self.save_every = save_every
        self.log_interval = log_interval
        self.grad_data = {}
        # Initialize the log file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({}, f)

    def log_gradients(self, epoch, model):
        """
        Logs the gradients of a PyTorch model every 'log_interval' epochs.

        Args:
            epoch (int): Current epoch number.
            model (torch.nn.Module): PyTorch model whose gradients should be logged.
        """
        if epoch % self.log_interval != 0:
            return  # Only log gradients at the specified interval

        epoch_key = f"epoch_{epoch}"
        self.grad_data[epoch_key] = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_values = param.grad.cpu().numpy().flatten()
                layer_name = name.replace(".weight", "").replace(".bias", " (bias)")

                self.grad_data[epoch_key][layer_name] = {
                    "mean": float(np.mean(grad_values)),
                    "median": float(np.median(grad_values)),
                    "max": float(np.max(grad_values)),
                    "min": float(np.min(grad_values))
                }

        # Save to JSON
        with open(self.log_path, "w") as f:
            json.dump(self.grad_data, f, indent=4)


    def get_gradient_data(self):
        """Returns the saved gradient data."""
        return self.grad_data
