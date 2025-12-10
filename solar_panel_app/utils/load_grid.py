import pandas as pd
import torch

def load_grid(csv_file):
    """
    Loads a panel-grid CSV used for anomaly forecasting.
    Returns:
        grid:   shape (1, H, W)
        mask:   shape (1, H, W) with 1 = valid cell, 0 = invalid
    """

    df = pd.read_csv(csv_file, header=None, dtype=str)

    def safe_convert(x):
        x = str(x).strip()
        if x in ["0", "1", "-1"]:
            return float(x)
        return -1.0

    arr = df.map(safe_convert).values.astype("float32")
    mask = (arr != -1).astype("float32")
    arr[arr == -1] = 0

    return (
        torch.tensor(arr).unsqueeze(0),
        torch.tensor(mask).unsqueeze(0)
    )
