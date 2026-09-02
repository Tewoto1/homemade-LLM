# Running a training job on a rented GPU

## 0. Before you rent anything (on your Mac)

```bash
cd ~/Desktop/LLMv1
python training/gradcheck.py     # must print "FAILURES: none" for tie=False and tie=True
git add -A && git commit -m "training loop, loss, checkpoints, weight tying"
git push                         # needs a remote: gh repo create LLMv1 --private --source=. --push
```

`gradcheck` is the gate. A broken backward costs you the whole rental.

## 1. On the box

```bash
ssh -p PORT root@HOST            # vast.ai gives a non-22 port; note it, scp needs it too
git clone YOUR_REPO_URL && cd LLMv1
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"
python training/gradcheck.py     # CPU + float64 by design, takes a second
```

## 2. Build the token file

```bash
python training/train.py --prepare roneneldan/TinyStories --out data/tiny.bin \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Prints the token count and the vocab to train against. TinyStories is ~500M tokens;
Chinchilla-optimal for a 100M model is ~2B, so a full run is roughly 4 epochs over it.

## 3. Find the batch size that fits

Activation memory is dominated by the logits, `batch * seq * vocab * 4` bytes, plus
roughly 400 MB per batch item across the blocks. Probe rather than guess:

```bash
for B in 8 16 32; do
  timeout 120 python training/train.py --data data/tiny.bin --config 100m-32k \
      --batch $B --steps 5 --log-every 1 2>&1 | tail -2
done
nvidia-smi
```

Take the largest that does not OOM, then back off one step.

## 4. Launch it detached

```bash
tmux new -s run
python training/train.py --data data/tiny.bin --config 100m-32k \
    --batch 16 --seq 512 --steps 60000 --ckpt-every 2000 2>&1 | tee train.log
# Ctrl-b then d to detach; tmux attach -t run to come back
```

**Use tmux.** Without it, an ssh drop kills the run and you pay for the whole night
having trained nothing.

The log line gives loss, perplexity, lr, gradient norm and tok/s. Set `--steps` from
the tok/s you see in the first minute, not from a guess. A gradient norm that climbs
steadily is the early warning that the lr is too high -- it moves well before the loss does.

## 5. Pull the checkpoint back (run on your Mac)

```bash
scp -P PORT root@HOST:~/LLMv1/checkpoints/final.pt ~/Desktop/LLMv1/checkpoints/
# mid-run, to grab whatever exists so far:
rsync -avP -e "ssh -p PORT" root@HOST:~/LLMv1/checkpoints/ ~/Desktop/LLMv1/checkpoints/
scp -P PORT root@HOST:~/LLMv1/train.log ~/Desktop/LLMv1/
```

A 100M fp32 checkpoint is ~400 MB. Verify it loads BEFORE you destroy the box:

```bash
python -c "
from loading import load_model
m, extra = load_model('checkpoints/final.pt', device='mps')
m.print(); print('step', extra['step'], '| final loss', extra['losses'][-1])
import torch
print(m.generate_KV_cache(torch.tensor([[464, 2933, 373]]), 40))
"
```

## 6. Kill the box

Destroy the instance in the provider's UI. Stopping is not destroying -- a stopped
vast.ai instance still bills for storage.

Checklist before you do: checkpoint copied, `train.log` copied, checkpoint loads on
the Mac, and anything you changed on the box is committed and pushed.
