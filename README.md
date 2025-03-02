# NeuralTrack

[NeuralTrack](https://github.com/sciencehope/NeuralTrack) is a lightweight logging tool for deep learning training, designed to track loss and gradient updates efficiently without slowing down the training process. It provides easy-to-use logging functionality and visualization tools to help monitor model performance.
## Features
✅ **Minimal Overhead** – Asynchronous logging ensures training speed is not compromised.  
✅ **Loss Tracking** – Logs individual loss components per epoch.  
✅ **Gradient Tracking** – Captures gradient statistics (mean, median, max, min) for each layer.  
✅ **Visualizations** – Generate loss and gradient plots with simple CLI commands.  

## Installation

Install NeuralTrack via pip:

```bash
pip install neuraltrack
```

## Quickstart
1️⃣ Logging Loss

In your training loop, use the LossLogger to track loss values:

```python
from neuraltrack.logging.loss_logger import LossLogger

logger = LossLogger("loss_log.json")

for epoch in range(num_epochs):
    LossLogger.start_epoch()
    
    for batch in dataloader:
        loss_dict = {"total_loss": loss_value, "reconstruction_loss": recon_loss}
        logger.add_batch_loss(loss_dict)
    
    logger.log_epoch_loss(epoch)
```

2️⃣ Logging Gradients

Track gradient statistics every few epochs:
```python
from neuraltrack.logging.gradient_logger import GradientLogger

grad_logger = GradientLogger("gradient_log.json", log_interval=10)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    loss.backward()
    grad_logger.log_gradients(epoch, model)
    optimizer.step()

```
## CLI Usage for LossPlotter and GradientPlotter
3️⃣ Plotting Loss

You can execute the loss plotting functionality from the CLI by running the following command in the terminal:
```bash
neuraltrack-plot-loss --log_path path_to_loss_log.json --show_plot --save_dir output_directory
```
Arguments for LossPlotter:
- `--log_path`: Path to the loss log JSON file (required).
- `--show_plot`: Optional flag to display the plot interactively (if this is passed, the plot will open in a window after saving it).
- `--save_dir`: Directory where the plot will be saved (optional; defaults to LossPlots).

Example Usage:
```bash
neuraltrack-plot-loss --log_path loss_log.json --show_plot
```
This command will generate a loss plot from the data in loss_log.json and display it interactively.

4️⃣ Plotting Gradients

To generate gradient plots, you can use the following command in your terminal:
```bash
neuraltrack-plot-gradient --log_path path_to_gradient_log.json --epoch 5 --show_plot --chart_type line --include_bias
```
Arguments for GradientPlotter:
- --log_path: Path to the gradient log JSON file (required).
- `--epoch`: Specific epoch number to plot (optional; default is 1).
- `--show_plot`: Optional flag to display the plot interactively (if this is passed, the plot will open in a window after saving it).
- `--chart_type`: The type of chart to use. Can be either line or bar (optional; default is line).
- `--include_bias`: Flag to include bias layers in the plot (optional; default is False).

Example Usage:
```bash
neuraltrack-plot-gradient --log_path gradient_log.json --epoch 5 --show_plot --chart_type bar --include_bias
```
This will generate a bar chart for the gradient values of epoch 5 and display the plot interactively.
<!-- ## CLI Commands
| Command | Description |
|---------|-------------|
| `neuraltrack-plot-loss --log_path loss_log.json` | Generates a loss plot |
| `neuraltrack-plot-gradient --log_path gradient_log.json --epoch 5` | Generates a gradient plot for epoch 5 |
| `--show_plot` | Displays the plot interactively |
| `--chart_type line` | Uses a line chart for gradients |
| `--include_bias` | Includes bias layers in gradient plots | -->

## Python Code Usage

Both LossPlotter and GradientPlotter can also be used directly within Python scripts.
### LossPlotter Example (Python)
```python
from neuraltrack.visualization.loss_plot import LossPlotter

# Initialize LossPlotter with the path to the log file and save directory
plotter = LossPlotter(log_path="loss_log.json", save_dir="LossPlots")

# Plot the losses (optionally show the plot)
plotter.plot_losses(show_plot=True)
```
- `log_path`: The path to your loss log JSON file.
- `save_dir`: The directory where the plot should be saved (optional; defaults to LossPlots).
- `show_plot`: If set to True, the plot will be shown interactively after it is generated.

### GradientPlotter Example (Python)
```python
from neuraltrack.visualization.gradient_plot import GradientPlotter

# Initialize GradientPlotter with your preferred arguments
plotter = GradientPlotter(
    log_path="gradient_log.json", 
    save_dir="GradientPlots", 
    chart_type="line", 
    include_bias=False
)

# Plot gradients for epoch 5
plotter.plot_gradients(show_plot=True, epoch=5)
```
- `log_path`: Path to your gradient log JSON file.
- `save_dir`: Directory where the gradient plot will be saved (optional; defaults to GradientPlots).
- `chart_type`: Choose between 'bar' or 'line' for chart type (optional; defaults to line).
- `show_plot`: If set to True, it will display the plot interactively.
- `epoch`: Specify which epoch's gradient data to plot.
- `include_bias`: Set to True if you want to include bias layers in the gradient plot.

### LossLogger and GradientLogger Parameters Explanation

Now let's explain the parameters for LossLogger and GradientLogger:
LossLogger Parameters:

- `log_path` (str): The path to the JSON file where the loss data will be logged (default: "loss_log.json").
- `epoch_start_time` (class variable): The time when the current epoch started (used internally to track epoch duration).
`loss_data` (dict): The dictionary where loss data is stored for each epoch.
- `loss_components` (dict): Holds the running loss components for each batch during an epoch (e.g., for multiple loss functions).
- `total_loss` (float): Tracks the total loss for the epoch (used internally).
- `batch_count` (int): The number of batches processed in the current epoch (used internally).
- `lock` (threading.Lock): Ensures thread safety when writing logs asynchronously.

### GradientLogger Parameters:

- `log_path` (str): The path to the JSON file where gradient data will be logged (default: "gradient_log.json").
- `save_every` (int): The number of epochs between saving gradient data to disk (default: 10).
- `log_interval` (int): The interval at which gradient data is logged (e.g., every 10 epochs).
- `grad_data` (dict): A dictionary where the gradient statistics for each epoch are stored.

## Roadmap

📌 Add support for additional metrics like accuracy and learning rate tracking.
📌 Extend visualization options (e.g., smoothing, multi-run comparisons).
## License

This project is licensed under the MIT License.

Let me know if you want any modifications! 🚀