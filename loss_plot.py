import re
import matplotlib.pyplot as plt

LOSS_FILE = "loss.txt"

def parse_loss_lr(path):
    """
    Parses HuggingFace Trainer log lines to extract:
    - training loss
    - learning rate
    - epoch
    """
    epochs = []
    losses = []
    lrs = []

    loss_pattern = re.compile(r"'loss': ([0-9\.eE+-]+)")
    lr_pattern = re.compile(r"'learning_rate': ([0-9\.eE+-]+)")
    epoch_pattern = re.compile(r"'epoch': ([0-9\.eE+-]+)")

    with open(path, "r") as f:
        for line in f:
            if "eval_loss" in line or "train_runtime" in line:
                continue

            loss_match = loss_pattern.search(line)
            lr_match = lr_pattern.search(line)
            epoch_match = epoch_pattern.search(line)

            # Only count training steps
            if loss_match and epoch_match and lr_match:
                epochs.append(float(epoch_match.group(1)))
                losses.append(float(loss_match.group(1)))
                lrs.append(float(lr_match.group(1)))

    return epochs, losses, lrs


def plot_loss_and_lr(epochs, losses, lrs):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ---- Loss curve (left axis) ----
    ax1.plot(epochs, losses, color="tab:blue", label="Training Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # ---- Learning Rate curve (right axis) ----
    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="tab:red", label="Learning Rate", linewidth=2, linestyle="--")
    ax2.set_ylabel("Learning Rate", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Title + Grid
    plt.title("Training Loss and Learning Rate vs Epoch", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Legends
    fig.tight_layout()
    plt.show()


def main():
    epochs, losses, lrs = parse_loss_lr(LOSS_FILE)
    print(f"Parsed {len(losses)} valid training entries.")
    plot_loss_and_lr(epochs, losses, lrs)


if __name__ == "__main__":
    main()
