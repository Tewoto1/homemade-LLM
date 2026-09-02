"""
Training loop. Everything is manual, so the loop is the whole story:

    logits     = model.forward(x)
    loss, grad = cross_entropy_loss(logits, y)   # grad is dL/dlogits
    model.zero_grad(); model.backward(grad)      # fills every self.grad_*
    clip_grad_norm(model, 1.0)
    model.step(lr)                               # W -= lr * grad_W

The shift-by-one lives in the data loader, not the loss: get_batch returns
x = tokens[i : i+s] and y = tokens[i+1 : i+s+1], so logit position j is already
lined up with the token that should follow it, and all s positions are supervised.

    python training/train.py --prepare roneneldan/TinyStories --out data/tiny.bin
    python training/train.py --data data/tiny.bin --steps 20000 --device cuda
    python training/train.py --smoke
"""

import os
import sys
import math
import time
import argparse

import torch    # pyright: ignore[reportMissingImports]

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import LLM                                             # noqa: E402
from loading import save_model, load_model, clip_grad_norm, to_device   # noqa: E402
from loss import cross_entropy_loss, IGNORE_INDEX                 # noqa: E402


# heads * attn_dim == hid_dim in all of these.
#
# Embedding + unembedding cost 2 * vocab * hid_dim and do not shrink as you add
# layers, so at these widths they dominate a small model. Two levers, both used
# below -- tie the head to the embedding table (halves it to one * vocab * hid_dim)
# and shrink the vocab:
#   V=50257 untied  d=512 L=16 -> 101.9M, 51.5M in embeddings (51%)
#   V=50257 tied    d=640 L=14 -> 101.1M, 32.2M               (32%)
#   V=32000 tied    d=640 L=16 ->  99.2M, 20.5M               (21%)
# The catch on vocab: llama's 32k needs ~14% more tokens than gpt2 for the same
# English, so you trade embedding parameters for sequence length.
CONFIG_50M = dict(layers = 10, hid_dim = 384, attn_dim = 64, heads = 6,
                  vocab_size = 50257, pos_embed = "cosine", max_s = 512)

# gpt2 tokenizer, tied
CONFIG_100M = dict(layers = 14, hid_dim = 640, attn_dim = 64, heads = 10,
                   vocab_size = 50257, pos_embed = "cosine", max_s = 512, tie = True)

# llama tokenizer, tied: --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
CONFIG_100M_32K = dict(layers = 16, hid_dim = 640, attn_dim = 64, heads = 10,
                       vocab_size = 32000, pos_embed = "cosine", max_s = 512, tie = True)

CONFIG_TINY = dict(layers = 2, hid_dim = 64, attn_dim = 16, heads = 4,
                   vocab_size = 512, pos_embed = "cosine", max_s = 128)

CONFIGS = {"tiny": CONFIG_TINY, "50m": CONFIG_50M,
           "100m": CONFIG_100M, "100m-32k": CONFIG_100M_32K}


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():   # Apple silicon
        return "mps"
    return "cpu"


class CosineScheduler():
    """Linear warmup then cosine decay. Warmup matters here: at step 0 the model is
    uniform over 50k tokens, loss ~10.8 nats, and the gradients are big enough to
    wreck LayerNorm's gamma if the full lr lands immediately."""

    def __init__(self, base_lr, warmup, total, min_lr = None):
        self.base_lr, self.warmup, self.total = base_lr, max(1, warmup), total
        self.min_lr = base_lr * 0.1 if min_lr is None else min_lr
        self.t = 0

    @property
    def lr(self):
        if self.t < self.warmup:
            return self.base_lr * (self.t + 1) / self.warmup
        if self.t >= self.total:
            return self.min_lr
        p = (self.t - self.warmup) / max(1, self.total - self.warmup)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * p))

    def step(self):
        self.t += 1


# ---------------------------------------------------------------- data

class PackedData():
    """One flat token stream, sampled at random offsets.

    Documents are concatenated with an EOS between them, so every batch is full
    length and there is NO padding at all. This is how GPT-2 style pretraining is
    normally done, and it is why padding barely matters until fine-tuning.
    """

    def __init__(self, tokens, seq_len, device = "cpu"):
        self.tokens, self.seq_len, self.device = tokens, seq_len, device

    def __len__(self):
        return len(self.tokens) - self.seq_len - 1

    def get_batch(self, batch_size):
        ix = torch.randint(len(self), (batch_size,)).tolist()
        x = torch.stack([torch.as_tensor(self.tokens[i : i + self.seq_len].astype("int64")) for i in ix])
        y = torch.stack([torch.as_tensor(self.tokens[i + 1 : i + 1 + self.seq_len].astype("int64")) for i in ix])
        return x.to(self.device), y.to(self.device)

    @classmethod
    def from_file(cls, path, seq_len, device = "cpu"):
        # memmap, not load: a pretraining .bin is routinely bigger than RAM, and we
        # only ever touch batch_size * seq_len of it at a time
        import numpy as np
        return cls(np.memmap(path, dtype = np.uint16, mode = "r"), seq_len, device)


class PaddedData():
    """Variable-length sequences, right-padded. This is where masking earns its keep.

    Attention needs no extra pad mask as long as padding is on the RIGHT: the causal
    mask already stops real tokens seeing anything after them, and the pad positions'
    own outputs are thrown away by the loss mask. Left-padding would break that.
    """

    def __init__(self, sequences, seq_len, pad_id = 0, device = "cpu"):
        self.sequences = [torch.as_tensor(s, dtype = torch.long) for s in sequences]
        self.seq_len, self.pad_id, self.device = seq_len, pad_id, device

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.sequences), (batch_size,)).tolist()
        xs, ys = [], []
        for i in ix:
            seq = self.sequences[i][: self.seq_len + 1]
            x, y = seq[:-1], seq[1:]
            pad = self.seq_len - len(x)
            if pad > 0:
                x = torch.cat([x, torch.full((pad,), self.pad_id, dtype = torch.long)])
                # IGNORE_INDEX, not pad_id: the model is never asked to predict padding
                y = torch.cat([y, torch.full((pad,), IGNORE_INDEX, dtype = torch.long)])
            xs.append(x); ys.append(y)
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)


def prepare(source, out_path, split = "train", text_field = None, limit = None,
            tokenizer = "gpt2"):
    """Tokenize a dataset into a flat uint16 .bin that PackedData.from_file reads.

    source is either a HuggingFace hub name ("roneneldan/TinyStories", "wikitext")
    or a local .json / .jsonl file. uint16 because the GPT-2 vocab is 50257 < 65536,
    which halves the file size against int32 for free.

    Streaming means you never materialize the dataset on disk, which matters for
    anything web-scale. It also means --limit is how you control the size.
    """
    import numpy as np
    from transformers import AutoTokenizer    # pyright: ignore[reportMissingImports]
    from datasets import load_dataset         # pyright: ignore[reportMissingImports]

    tok = AutoTokenizer.from_pretrained(tokenizer)
    eos = tok.eos_token_id
    if len(tok) > 65535:
        raise SystemExit(f"vocab {len(tok):,} does not fit in uint16")

    if source.endswith(".json") or source.endswith(".jsonl"):
        ds = load_dataset("json", data_files = source, split = "train", streaming = True)
    else:
        ds = load_dataset(source, split = split, streaming = True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok = True)
    total, docs = 0, 0
    with open(out_path, "wb") as f:
        for row in ds:
            field = text_field or next(k for k, v in row.items() if isinstance(v, str))
            # add_special_tokens=False matters outside gpt2: llama-family tokenizers
            # prepend a BOS to every call, which would put one mid-stream on every doc
            ids = tok(row[field], add_special_tokens = False)["input_ids"] + [eos]
            np.array(ids, dtype = np.uint16).tofile(f)
            total += len(ids); docs += 1
            if docs % 5000 == 0:
                print(f"  {docs:,} docs  {total:,} tokens", flush = True)
            if limit and total >= limit:
                break
    print(f"{docs:,} docs -> {total:,} tokens -> {out_path} ({total * 2 / 1e9:.2f} GB)")
    print(f"vocab {len(tok):,} -- train with a matching --config, or --vocab {len(tok)}")


# ---------------------------------------------------------------- trainer

class Trainer():
    def __init__(self, model, data, scheduler, optimizer = None, grad_clip = 1.0,
                 ckpt_dir = "checkpoints"):
        self.model, self.data, self.sched = model, data, scheduler
        # optimizer: anything with .step(lr) reading the model's grad_* attrs.
        # None means plain SGD via model.step(lr). loading.named_slots() gives the
        # (name, owner, param_attr, grad_attr) pairing a stateful optimizer needs.
        self.optimizer = optimizer
        self.grad_clip, self.ckpt_dir = grad_clip, ckpt_dir
        self.losses, self.step_count = [], 0

    def train_single_step(self, x, y):
        loss, grad_logits, n_valid = cross_entropy_loss(self.model.forward(x), y)
        self.model.zero_grad()
        self.model.backward(grad_logits)        # head -> ln_f -> blocks -> embed
        norm = clip_grad_norm(self.model, self.grad_clip)
        (self.model if self.optimizer is None else self.optimizer).step(self.sched.lr)
        self.sched.step()
        self.step_count += 1
        self.losses.append(float(loss))
        return float(loss), norm, n_valid

    def train(self, steps, batch_size, log_every = 10, ckpt_every = 0):
        t0 = time.time()
        for _ in range(steps):
            x, y = self.data.get_batch(batch_size)
            loss, norm, n_valid = self.train_single_step(x, y)
            if log_every and self.step_count % log_every == 0:
                per_step = (time.time() - t0) / max(1, self.step_count)
                print(f"step {self.step_count:>6} | loss {loss:7.4f} | ppl {math.exp(min(loss, 20)):>9.2f} "
                      f"| lr {self.sched.lr:.2e} | grad {norm:7.3f} "
                      f"| {per_step:.3f}s/step | {n_valid * batch_size / per_step / batch_size:,.0f} tok/s",
                      flush = True)
            if ckpt_every and self.step_count % ckpt_every == 0:
                self.save(os.path.join(self.ckpt_dir, f"step_{self.step_count}.pt"))
        return self.losses

    def evaluate(self, batches = 20, batch_size = 8):
        total = sum(float(cross_entropy_loss(self.model.forward(x), y)[0])
                    for x, y in (self.data.get_batch(batch_size) for _ in range(batches)))
        mean = total / batches
        return mean, math.exp(min(mean, 20))

    def save(self, path):
        save_model(self.model, path, step = self.step_count, losses = self.losses,
                   sched_t = self.sched.t)
        print(f"saved {path}", flush = True)
        return path

    @classmethod
    def resume(cls, path, data, scheduler, device = "cpu", **kw):
        model, extra = load_model(path, device = device)
        t = cls(model, data, scheduler, **kw)
        t.step_count = extra.get("step", 0)
        t.losses = extra.get("losses", [])
        scheduler.t = extra.get("sched_t", t.step_count)
        print(f"resumed {path} at step {t.step_count}")
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action = "store_true", help = "tiny model + synthetic data")
    ap.add_argument("--prepare", type = str, help = "HF dataset name or .json/.jsonl to tokenize")
    ap.add_argument("--out", type = str, default = "data/tokens.bin")
    ap.add_argument("--text-field", type = str, default = None)
    ap.add_argument("--limit", type = int, default = None, help = "stop after N tokens")
    ap.add_argument("--data", type = str, help = ".bin of uint16 token ids")
    ap.add_argument("--device", type = str, default = default_device())
    ap.add_argument("--config", type = str, default = "100m", choices = list(CONFIGS))
    ap.add_argument("--tokenizer", type = str, default = "gpt2")
    ap.add_argument("--vocab", type = int, default = None, help = "override the config vocab_size")
    ap.add_argument("--no-tie", action = "store_true", help = "give the head its own weight matrix")
    ap.add_argument("--steps", type = int, default = 20000)
    ap.add_argument("--batch", type = int, default = 32)
    ap.add_argument("--seq", type = int, default = 512)
    ap.add_argument("--lr", type = float, default = 0.3, help = "tuned for plain SGD; ~1e-3 for AdamW")
    ap.add_argument("--warmup", type = int, default = 200)
    ap.add_argument("--resume", type = str, default = None)
    ap.add_argument("--ckpt-every", type = int, default = 1000)
    ap.add_argument("--log-every", type = int, default = 10)
    args = ap.parse_args()

    if args.prepare:
        return prepare(args.prepare, args.out, text_field = args.text_field,
                       limit = args.limit, tokenizer = args.tokenizer)

    # Nothing here builds an autograd graph, so torch should not either
    torch.set_grad_enabled(False)

    if args.smoke:
        torch.manual_seed(0)
        model = LLM(**CONFIG_TINY)
        model.print()
        data = PackedData(torch.arange(64).repeat(400).numpy().astype("uint16"), seq_len = 32)
        # plain SGD, so the lr is ~1000x an Adam default. At 1e-2 this barely moves.
        trainer = Trainer(model, data, CosineScheduler(0.3, warmup = 20, total = 400))
        print(f"random init should sit near ln(512) = {math.log(512):.2f}")
        trainer.train(steps = 400, batch_size = 8, log_every = 100)
        print("eval loss %.4f | ppl %.2f" % trainer.evaluate(5, 8))
        return

    if not args.data:
        raise SystemExit("need --data (or --smoke, or --prepare)")

    data = PackedData.from_file(args.data, seq_len = args.seq, device = args.device)
    sched = CosineScheduler(args.lr, warmup = args.warmup, total = args.steps)

    if args.resume:
        trainer = Trainer.resume(args.resume, data, sched, device = args.device)
    else:
        cfg = dict(CONFIGS[args.config], max_s = args.seq)
        if args.vocab:
            cfg["vocab_size"] = args.vocab
        if args.no_tie:
            cfg["tie"] = False
        model = to_device(LLM(**cfg), args.device)
        model.print()
        trainer = Trainer(model, data, sched)

    print(f"training on {args.device}, {len(data):,} positions available")
    trainer.train(args.steps, args.batch, log_every = args.log_every, ckpt_every = args.ckpt_every)
    trainer.save(os.path.join(trainer.ckpt_dir, "final.pt"))


if __name__ == "__main__":
    main()
