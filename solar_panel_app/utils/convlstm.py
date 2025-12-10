import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )
        self.hidden_channels = hidden_channels

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch, H, W):
        return (torch.zeros(batch, self.hidden_channels, H, W),
                torch.zeros(batch, self.hidden_channels, H, W))

# -----------------------------------------------
# Utility to load & process anomaly grids
# -----------------------------------------------
def load_grid(path):
    df = pd.read_csv(path, header=None, dtype=str)

    # Convert numeric-like strings to values
    def safe(x):
        x = str(x).strip()
        if x in ["0", "1", "-1"]:
            return float(x)
        return -1.0  # treat unexpected values as no panel

    arr = df.map(safe).values.astype("float32")

    # Replace -1 (no panel) with 0 for training, but we'll mask it later
    mask = (arr != -1).astype("float32")   # 1 = real panel, 0 = no panel
    arr[arr == -1] = 0

    tensor = torch.tensor(arr).unsqueeze(0)   # (1, H, W)
    mask = torch.tensor(mask).unsqueeze(0)   # (1, H, W)
    return tensor, mask


# -----------------------------------------------
# ConvLSTM Model
# -----------------------------------------------
class ConvLSTMModel(nn.Module):
    def __init__(self, in_ch=1, hidden=16):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden)
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(self, seq):
        # seq shape: (T,1,H,W)
        T, _, H, W = seq.shape
        h, c = self.cell.init_state(1, H, W)

        for t in range(T):
            h, c = self.cell(seq[t].unsqueeze(0), (h, c))

        return self.out(h)  # logits