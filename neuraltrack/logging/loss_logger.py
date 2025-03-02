import threading
import json
import time
import os
import torch

class LossLogger:
    epoch_start_time = None  # Store start time of current epoch

    def __init__(self, log_path="loss_log.json"):
        self.log_path = log_path
        self.start_time = time.time()
        self.loss_data = {}
        self.loss_components = {}  # Store running loss sums
        self.total_loss = 0
        self.batch_count = 0
        self.lock = threading.Lock()  # Ensure thread safety

        # Initialize the log file if it doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump({}, f)

    @classmethod
    def start_epoch(cls):
        cls.epoch_start_time = time.time()
        cls.total_loss = 0
        cls.batch_count = 0
        cls.loss_components = {}

    def add_batch_loss(self, batch_loss_dict):
        """
        Adds batch loss values.

        Args:
            batch_loss_dict (dict): Losses for a batch.
        """
        self.batch_count += 1
        for key, value in batch_loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            if key not in self.loss_components:
                self.loss_components[key] = 0
            self.loss_components[key] += value

    def log_epoch_loss(self, epoch):
        """
        Computes and logs the average loss for the epoch.
        """
        if LossLogger.epoch_start_time is None:
            raise ValueError("Epoch has not been started. Call LossLogger.start_epoch() first.")

        epoch_key = f"epoch_{epoch}"
        time_for_epoch = time.time() - LossLogger.epoch_start_time
        total_time_elapsed = time.time() - self.start_time

        # Compute average loss per batch
        avg_loss_dict = {
            key: value / self.batch_count
            for key, value in self.loss_components.items()
        }

        # Store loss data
        self.loss_data[epoch_key] = {
            "time_for_epoch": time_for_epoch,
            "total_time_elapsed": total_time_elapsed,
            **avg_loss_dict,
        }

        # Write JSON in a separate thread
        thread = threading.Thread(target=self._save_log)
        thread.start()

    def _save_log(self):
        """Write loss data to JSON asynchronously."""
        with self.lock:
            with open(self.log_path, "w") as f:
                json.dump(self.loss_data, f, indent=4)
