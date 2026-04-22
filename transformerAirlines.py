import os
import math
from typing import Optional, Tuple

from matplotlib.style import context
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn




class Attention(nn.Module):
    def __init__(self, d: int, k: int):
        super(Attention, self).__init__()
        self.Wq, self.Wk, self.Wv = self.init_qkv(d, k)
        self.Wo = nn.Parameter(torch.randn(k, d))

    def init_qkv(self, d: int, k: int):
        Wq, Wk, Wv = [nn.Parameter(torch.randn(d, k)) for _ in range(3)]
        return Wq, Wk, Wv

    def get_qkv(self, X: torch.Tensor):
        return X @ self.Wq, X @ self.Wk, X @ self.Wv

    def self_attention(self, X: torch.Tensor) -> torch.Tensor:
        Q, K, V = self.get_qkv(X)
        return self.forward(Q, K, V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        d_k = K.shape[1]
        sim = Q @ K.mT    #permute(0,2,1) instead of mT from tensor
        att_weights = torch.softmax(sim / math.sqrt(d_k), dim=-1)
        out = att_weights @ V
        out = out @ self.Wo
        return out

    def cross_attention(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        Q, _, _ = self.get_qkv(X)
        _, K, V = self.get_qkv(Y)
        return self.forward(Q, K, V)


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(max_len, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)
        return x + pe


class EncoderBlock(nn.Module):
    def __init__(self, d: int, k: int):
        super(EncoderBlock, self).__init__()
        self.attn = Attention(d, k)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.ReLU(), nn.Linear(d * 4, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_out = self.attn.self_attention(x)
        x = self.norm1(x + att_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
class Encoder(nn.Module):
    def __init__(self, d: int, k: int, N: int):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d, k) for _ in range(N)])
        self.pos_enc = PositionalEncoding(d)
    def forward(self, x):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d: int, k: int):
        super(DecoderBlock, self).__init__()
        self.self_attn = Attention(d, k)
        self.cross_attn = Attention(d, k)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 4),
            nn.ReLU(),
            nn.Linear(d * 4, d)
            )
    def forward(self, x, enc_out):
        self_attn_out = self.self_attn.self_attention(x)
        x = self.norm1(x + self_attn_out)
        cross_attn_out = self.cross_attn.cross_attention(x, enc_out)
        x = self.norm2(x + cross_attn_out)
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x

class Decoder(nn.Module):
    def __init__(self, d: int, k: int, N: int):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d, k) for _ in range(N)
        ])
        self.pos_enc = PositionalEncoding(d)
    def forward(self, x, enc_out):
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out)
        return x

class Transformer(nn.Module):
    def __init__(self, d: int, k: int, N: int):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d, k, N)
        self.decoder = Decoder(d, k, N)
    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out)
        return dec_out
    def generate(self, src, max_len=20):
        enc_out = self.encoder(src)
        dec_input = torch.zeros(src.size(0), 1, enc_out.size(-1)).to(src.device)
        generated = []
        for _ in range(max_len):
            dec_out = self.decoder(dec_input, enc_out)
            output = dec_out[:, -1, :].unsqueeze(1)
            generated.append(output)
            dec_input = torch.cat([dec_input, output], dim=1)
        return torch.stack(generated, dim=1)
class Embedder(nn.Module):
    def __init__(self, d):
        super(Embedder, self).__init__()
        self.w_in = nn.Parameter(torch.randn(1, d))
        self.b_in = nn.Parameter(torch.zeros(d))
# initialize so that (x * w_in) @ w_out = x
        self.w_out = nn.Parameter(self.w_in / (self.w_in.norm()+ 1e-8))
        self.b_out = nn.Parameter(torch.zeros(1))
    def embed(self, x):
        return x * self.w_in + self.b_in
    def unembed(self, x):
        return x @ self.w_out.T + self.b_out

class OutputScaler(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.scale = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Linear(embed_dim, embed_dim)

    def forward(self, X, context):
        head_in = X[:, -1, :]
        out = self.head(head_in)
        s = torch.sigmoid(self.scale(context))
        b = self.bias(context)
        return out * s + b
class TimeSeriesTransformer(nn.Module):
    def __init__(self, d, k, N):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = Transformer(d, k, N)
        self.embedder, self.scaler = Embedder(d), OutputScaler(d)
    def forward(self, src, tgt):
        src_emb, tgt_emb = self.embedder.embed(src), self.embedder.embed(tgt)
        transformer_out = self.transformer(src_emb, tgt_emb)
        scaled_out = self.scaler(transformer_out, src_emb.mean(dim=1))
        return self.embedder.unembed(scaled_out)
    def generate(self, src, max_len=20):
        src_emb = self.embedder.embed(src)
        transformer_out = self.transformer.generate(src_emb, max_len)
        # shape: (batch, max_len, 1, d), squeeze to (batch, max_len, d)
        transformer_out = transformer_out.squeeze(2)
        # Reshape to (batch * max_len, d)
        batch_size = transformer_out.size(0)
        transformer_out_flat = transformer_out.view(-1, transformer_out.size(-1))
        # Unembed to get (batch * max_len, 1)
        output_flat = self.embedder.unembed(transformer_out_flat)
        # Reshape back to (batch, max_len)
        output = output_flat.view(batch_size, max_len)
        return output

def plot_series(train_series, test_series=None, pred_series=None, title: str = "", m: str = "o"):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(train_series, label="Train", color="blue", marker=m)
    if test_series is not None:
        test_x = np.arange(len(train_series), len(train_series) + len(test_series))
        plt.plot(test_x, test_series, label="Test", color="orange", marker=m)
    if pred_series is not None:
        pred_x = np.arange(len(train_series), len(train_series) + len(pred_series))
        plt.plot(pred_x, pred_series, label="Prediction", color="green", marker=m)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show(block = False)


def get_airline_passenger_data() -> np.ndarray:
    flights = sns.load_dataset("flights")
    series = flights.pivot(index="year", columns="month", values="passengers")
    return series.values.flatten()


def normalize_series(train_series: np.ndarray, test_series: Optional[np.ndarray] = None):
    s_max = np.max(train_series)
    s_min = np.min(train_series)
    train_series = (train_series - s_min) / (s_max - s_min)
    if test_series is not None:
        test_series = (test_series - s_min) / (s_max - s_min)
    return train_series, test_series


def train_test_split(series: np.ndarray, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    split_idx = int(len(series) * train_size)
    return series[:split_idx], series[split_idx:]


def create_dataset(series, input_len: int, output_len: int):
    X, Y = [], []
    for i in range(len(series) - input_len - output_len + 1):
        X.append(series[i : i + input_len])
        Y.append(series[i + input_len : i + input_len + output_len])
    return torch.utils.data.TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
    )


def get_datasets(in_len: int = 12, out_len: int = 6):
    series = get_airline_passenger_data()
    train_series, test_series = train_test_split(series)
    train_series, test_series = normalize_series(train_series, test_series)
    train_ds = create_dataset(train_series, in_len, out_len)
    test_ds = create_dataset(test_series, in_len, out_len)
    return train_ds, test_ds


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(42)


def plot_losses(train_loss, test_loss):
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label="Train Loss", color="blue")
    plt.plot(test_loss, label="Test Loss", color="orange")
    plt.title("Training and Test Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show(block = False)


def train_loop(model, train_ds, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X, Y in train_ds:
        optimizer.zero_grad()
        output = model.generate(X.unsqueeze(-1), max_len=Y.size(1))
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_ds)


def test_loop(model, test_ds, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, Y in test_ds:
            output = model.generate(X.unsqueeze(-1), max_len=Y.size(1))
            loss = criterion(output, Y)
            total_loss += loss.item()
    return total_loss / len(test_ds)


def train_model(model, train_ds, test_ds, epochs: int = 400, lr: float = 5e-4, batch_size: int = 8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    test_x, test_y = test_ds.tensors

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    train_loss, test_loss = [], []
    for epoch in range(epochs):
        avg_loss = train_loop(model, train_loader, optimizer, criterion)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")

        avg_loss = test_loop(model, test_loader, criterion)
        test_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {avg_loss:.4f}")

        if (epoch + 1) % 100 == 0:
            forecast = (
                model.generate(test_x[0].unsqueeze(0).unsqueeze(-1), max_len=6).squeeze(0).detach().numpy()
            )
            plot_series(test_x[0].numpy(), test_y[0].numpy(), forecast, title=f"Epoch {epoch+1}")

            plot_losses(train_loss, test_loss)
            lr /= 2


def _init_panel(ax, split_name, x_data, in_len, out_len):
    m = "o" if split_name == "Test" else None
    panel = {
        "ctx": ax.plot([], [], color="steelblue", marker=m, linewidth=2, label="Context")[0],
        "act": ax.plot([], [], color="darkorange", marker=m, linewidth=2, label="Actual Future")[0],
        "pred": ax.plot([], [], color="seagreen", marker=m, linewidth=2, label="Forecast")[0],
        "cut": ax.axvline(0, color="gray", linestyle=":", linewidth=1.5),
        "txt": ax.text(
            0.02,
            0.96,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "lightgray"},
        ),
    }
    split_len = len(x_data) + in_len + out_len - 1
    ax.set_xlim(0, split_len - 1)
    ax.set_title(f"{split_name} Split")
    ax.set_xlabel("Actual Timestep")
    ax.grid(alpha=0.25)
    return panel


def _update_panel(model, ax, split_name, x_data, y_data, panel, frame_idx, in_len, out_len):
    i = frame_idx % len(x_data)
    x_seq, y_true = x_data[i], y_data[i]
    with torch.no_grad():
        y_pred = model.generate(x_seq.unsqueeze(0).unsqueeze(-1), max_len=out_len).squeeze(0)
    ctx_t = list(range(i, i + in_len))
    fut_t = list(range(i + in_len, i + in_len + out_len))
    y_pred_np = y_pred.detach().numpy()
    panel["ctx"].set_data(ctx_t, x_seq.numpy())
    panel["act"].set_data(fut_t, y_true.numpy())
    panel["pred"].set_data(fut_t, y_pred_np)
    panel["cut"].set_xdata([i + in_len - 0.5, i + in_len - 0.5])
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    panel["txt"].set_text(f"sample {i + 1}/{len(x_data)}\nMAE: {mae:.4f}")
    ax.set_title(f"{split_name} Split")
    vals = x_seq.numpy().tolist() + y_true.numpy().tolist() + y_pred_np.tolist()
    artists = [panel["ctx"], panel["act"], panel["pred"], panel["cut"], panel["txt"]]
    return vals, artists


def plot_sliding_window_predictions(model, train_ds, test_ds):
    train_x, train_y = train_ds.tensors
    test_x, test_y = test_ds.tensors
    in_len, out_len = train_x.size(1), train_y.size(1)
    splits = [("Train", train_x, train_y), ("Test", test_x, test_y)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    panels = {name: _init_panel(ax, name, x_data, in_len, out_len) for ax, (name, x_data, _) in zip(axes, splits)}

    def update(frame_idx):
        vals, updated = [], []
        for ax, (name, x_data, y_data) in zip(axes, splits):
            panel_vals, panel_artists = _update_panel(
                model, ax, name, x_data, y_data, panels[name], frame_idx, in_len, out_len
            )
            vals.extend(panel_vals)
            updated.extend(panel_artists)
        y_min, y_max = min(vals), max(vals)
        pad = max(0.05, (y_max - y_min) * 0.15)
        for ax in axes:
            ax.set_ylim(y_min - pad, y_max + pad)
        return updated

    frames = max(len(train_x), len(test_x))
    anim = FuncAnimation(fig, update, frames=frames, interval=850, blit=False, repeat=True)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    plt.show()

def forecast_sequence(model, x_seq, out_len=None) -> np.ndarray:
    if out_len is None:
        out_len = x_seq.size(0) // 2

    model.eval()
    context = x_seq.detach().clone()
    preds = []
    with torch.no_grad():
        remaining = out_len
        while remaining > 0:
            step = remaining
            y_chunk = model.generate(context.unsqueeze(0).unsqueeze(-1), max_len=step).squeeze(0)
            preds.append(y_chunk)
            context = torch.cat([context, y_chunk], dim=0)[-x_seq.size(0):]
            remaining -= step

    if len(preds) == 0:
        return np.array([])
    return torch.cat(preds, dim=0).detach().cpu().numpy()
def forecast_all_inputs(model, input_len: int = 12, out_len: int = 6, train_size: float = 0.8):
    series = get_airline_passenger_data()
    train_series, test_series = train_test_split(series, train_size=train_size)
    train_series, test_series = normalize_series(train_series, test_series)
    context = torch.tensor(train_series[-input_len:], dtype=torch.float32)
    forecast_parts = []
    remaining = len(test_series)
    while remaining > 0:
        step = min(out_len, remaining)
        forecast_chunk = forecast_sequence(model, context, out_len=step)
        forecast_parts.append(forecast_chunk)
        context = torch.cat([
            context, torch.tensor(forecast_chunk, dtype=torch.float32)
        ], dim=0)[-input_len:]
        remaining -= step
    forecast = np.concatenate(forecast_parts)
    mae = float(np.mean(np.abs(forecast - test_series)))
    plot_series(train_series, test_series, forecast, title=f"Forecast (MAE={mae:.4f})", m="")


if __name__ == "__main__":
    train_ds, test_ds = get_datasets()
    # TimeSeriesTransformer must be defined/imported elsewhere in the project
    model = TimeSeriesTransformer(d=6, k=12, N=1)
    train_model(model, train_ds, test_ds)
    model.eval()
    #plot_sliding_window_predictions(model, train_ds, test_ds)
    forecast_all_inputs(model)
    plt.show(block=True)
    #torch.save(model.state_dict(), "timeseries_transformer.pth")
