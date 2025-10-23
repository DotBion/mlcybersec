import time, torch, torch.nn as nn, torch.nn.functional as F, pandas as pd

def has_mps():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def sync(dev):
    if dev == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif dev == "mps" and has_mps():
        import torch.mps; torch.mps.synchronize()

def time_it(label, device, run_fn, warmup=2, iters=8):
    for _ in range(warmup): run_fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters): run_fn()
    sync(device)
    t1 = time.perf_counter()
    return {"test": label, "device": device, "iters": iters,
            "total_sec": t1 - t0, "avg_ms_per_iter": (t1 - t0)*1000/iters}

# --- Matmul 512x512 ---
def make_matmul(device):
    n = 512
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    def run():
        c = a @ b
        _ = c[0,0]
    return run

# --- Tiny CNN forward ---
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8,16, 3, padding=1)
        self.fc = nn.Linear(16*16*16, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)        # 32x32 -> 16x16
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (16,16))
        x = x.view(x.size(0), -1)
        return self.fc(x)

def make_cnn_run(device):
    model = TinyCNN().to(device)
    x = torch.randn(8, 3, 32, 32, device=device)
    def run():
        with torch.no_grad():
            y = model(x); _ = y[0,0]
    return run

devices = ["cpu"] + (["mps"] if has_mps() else [])
rows = []
for dev in devices:
    rows.append(time_it("matmul_512x512", dev, make_matmul(dev), warmup=2, iters=6))
    rows.append(time_it("tinycnn_forward_8x3x32x32", dev, make_cnn_run(dev), warmup=2, iters=10))

print(pd.DataFrame(rows))
