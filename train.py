import os
import time
import argparse
import torch
from model import TinyLM, ModelConfig, count_params
import torch_directml

#training algorithm, developed 24.02.2026


parser = argparse.ArgumentParser(description="Train TinyLM")
parser.add_argument("--data",       type=str,   default="data/training.txt")
parser.add_argument("--steps",      type=int,   default=3000)
parser.add_argument("--batch",      type=int,   default=32)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--eval-every", type=int,   default=200)
parser.add_argument("--save-path",  type=str,   default="checkpoints/model.pt")
args = parser.parse_args()


#device selection
try:
    device = torch_directml.device()
    print("DirectML")
except Exception:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Applу")
    else:
        device = torch.device("cpu")
        print("No GPU found")


#Load training data from training.txt

print(f"\nLoading data from: {args.data}")
with open(args.data, "r", encoding="utf-8") as f:
    text = f.read()

#Build vocabulary from the data
chars    = sorted(set(text))
vocab    = {ch: i for i, ch in enumerate(chars)}
inv_vocab = {i: ch for ch, i in vocab.items()}

encode = lambda s: [vocab.get(c, 0) for c in s]
decode = lambda l: "".join(inv_vocab.get(i, "?") for i in l)

data   = torch.tensor(encode(text), dtype=torch.long)
n      = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

#Model

cfg = ModelConfig()
cfg.vocab_size = len(chars)

model = TinyLM(cfg).to(device)
print(f"\nModel: {count_params(model)} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

os.makedirs("checkpoints", exist_ok=True)

#sampler

def get_batch(split: str):
    src = train_data if split == "train" else val_data
    ix  = torch.randint(len(src) - cfg.context_len, (args.batch,))
    x   = torch.stack([src[i:i + cfg.context_len]     for i in ix]).to(device)
    y   = torch.stack([src[i + 1:i + cfg.context_len + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def evaluate(eval_iters: int = 100):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()
    return losses

best_val_loss = float("inf")
t0 = time.time()

for step in range(args.steps):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % args.eval_every == 0 or step == args.steps - 1:
        losses = evaluate()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(
            f"step {step:5d}/{args.steps}"
            f"train loss {losses['train']:.4f} "
            f"val loss {losses['val']:.4f} "
            f"lr {lr_now:.2e} "
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "vocab": vocab,
                    "inv_vocab": inv_vocab,
                    "val_loss": best_val_loss,
                    "step": step,
                },
                args.save_path,
            )
            print(f"Saved best model (val loss: {best_val_loss:.4f})")

total_time = time.time() - t0
print(f"   Best val loss: {best_val_loss:.4f}")
print(f"   Checkpoint saved to: {args.save_path}"
