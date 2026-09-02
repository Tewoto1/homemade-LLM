"""
Building an LLM from scratch v1, simple, like 2016 after attention is all you need version
"""
import torch             # pyright: ignore[reportMissingImports] FUCK YOU PYRIGHT
from torch import nn     # pyright: ignore[reportMissingImports]
from itertools import reduce
# Using nn for the Params class, nice for keeping track of things

# First defining linear layers and attention layers
# Part of the challenge is to implement autograd

class linear():
    def __init__(self, input_dim, output_dim = None):
        self.input_dim = input_dim
        if output_dim:
            self.output_dim = output_dim
        else:
            self.output_dim = input_dim
        self.W = nn.Parameter(torch.normal(0, (1/(2*input_dim)) ** 0.5, size = (input_dim, self.output_dim)))
        self.b = nn.Parameter(torch.zeros(self.output_dim))
        self.x = torch.zeros(1, 1, input_dim)  # Placeholder for input, will be set in forward
        # gradients
        self.grad_W = torch.zeros_like(self.W)
        self.grad_b = torch.zeros_like(self.b)
    
    def forward(self, x: torch.Tensor):
        # Do not call before backward
        # x of shape (batch_size, seq_len, input_dim)
        self.x = x
        return torch.matmul(x, self.W) + self.b

    def backward(self, grad_o: torch.Tensor):
        # grad_o of shape (batch_size, seq_len, output_dim)
        # Since o = Wx + b, df/dx_j = \sum df/do_i * do_i/dx_j = df/do * W_*j, so do/dx = grad_o * W^T
        grad_x = torch.matmul(grad_o, self.W.T) # shape (batch_size, seq_len, input_dim)
        # df/dW_ij = df/do_i * do_i/dW_ij and since o_i = \sum_j W_ij * x_j + b_i, do_i/dW_ij = x_j, so df/dW_ij = df/do_i * x_j
        # Keeping track of dim, self.x = (batch_size, seq_len, input_dim), grad_o = (batch_size, seq_len, output_dim), so grad_W = (input_dim, output_dim)
        self.grad_W = torch.matmul(self.x.transpose(-2, -1), grad_o) # shape (batch_size, input_dim, output_dim)
        self.grad_W = torch.sum(self.grad_W, dim = 0) # Sum over batch and sequence
        self.grad_b = torch.sum(grad_o, dim = (0, 1)) # Sum over batch and sequence
        return grad_x

    def step(self, lr):
        self.W.data -= lr * self.grad_W
        self.b.data -= lr * self.grad_b


    def zero_grad(self):
        self.grad_W = torch.zeros_like(self.W)
        self.grad_b = torch.zeros_like(self.b)

    def params(self):
        return [self.W, self.b]


# attention
class Attention():
    # Attention has Q, K, V matrices and it's chill
    # The formula is softmax(mask(QK^T/sqrt(hidden_dim))V)
    def __init__(self, input_dim, hidden_dim, num_heads):
        self.input_dim = input_dim
        self.hidden = hidden_dim
        self.heads = num_heads
        self.W_Q = nn.Parameter(torch.normal(0, (1/(2*input_dim)) ** 0.5, size = (input_dim, hidden_dim * num_heads)))
        # After Q_proj it's of shape (b, s, hid * heads), QK^T is (b, s, s)
        # After softmax it's (b, s, s), and V is (b, s, out)
        self.W_K = nn.Parameter(torch.normal(0, (1/(2*input_dim)) ** 0.5, size = (input_dim,hidden_dim * num_heads)))
        self.W_V = nn.Parameter(torch.normal(0, (1/(2*input_dim)) ** 0.5, size = (input_dim, hidden_dim * num_heads)))
        self.W_O = nn.Parameter(torch.normal(0, (1/(2*(hidden_dim * num_heads))) ** 0.5, size = (hidden_dim * num_heads, input_dim)))
        # Intermediate vars for calculation of grads
        self.x = torch.zeros(1, 1, input_dim)
        self.attn = torch.zeros(1, 1, 1, 1) # Placeholder for attention weights
        self.almost = torch.zeros(1, 1, hidden_dim * num_heads) # Placeholder for the output of attention before W_O
        self.Q = torch.zeros(1, num_heads, 1, hidden_dim)
        self.K = torch.zeros(1, num_heads, 1, hidden_dim)
        self.V = torch.zeros(1, num_heads, 1, input_dim)
        # gradients
        self.grad_W_Q = torch.zeros_like(self.W_Q)
        self.grad_W_K = torch.zeros_like(self.W_K)
        self.grad_W_V = torch.zeros_like(self.W_V)
        self.grad_W_O = torch.zeros_like(self.W_O)

    def forward(self, x: torch.Tensor, mask: bool = True):
        # DO NOT DO BEFORE BACKWARD
        # x is of shape (batch_size, seq_len, input_dim)
        b, s, _ = x.shape
        self.x = x
        self.Q = torch.matmul(x, self.W_Q).view(b, s, self.heads, self.hidden).permute(0, 2, 1, 3) # shape (b, h, s, d)
        self.K = torch.matmul(x, self.W_K).view(b, s, self.heads, self.hidden).permute(0, 2, 1, 3) # shape (b, h, s, d)
        self.V = torch.matmul(x, self.W_V).view(b, s, self.heads, self.hidden).permute(0, 2, 1, 3) # shape (b, h, s, d) id for input_dim
        mask = torch.tril(torch.ones(s, s, device=x.device))
        if mask is False:
            mask = torch.ones(s, s, device=x.device)
        scores = torch.matmul(self.Q, self.K.transpose(-2, -1)) / (self.hidden ** 0.5)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        self.attn = torch.softmax(scores, dim = -1)
        # attn of shape (b, h, s, s), V of shape (b, h, s, d), so multiply we get (b, h, s, d)
        self.almost = torch.matmul(self.attn, self.V).permute(0, 2, 1, 3).contiguous().view(b, s, self.heads * self.hidden).contiguous() # shape (b, s, h*d)
        output = torch.matmul(self.almost, self.W_O) # shape (b, s, id)
        # In the actual forward pass, we add the attention to x as a residual connection
        return output

    def __call__(self, x):
        return self.forward(x)

    def forward_KV_cache(self, x: torch.Tensor, mask: bool = False):
        # Under not RoPE, if we extend past max token length we would have to recalculate all the KV again, since their pos embedding changes
        # Assuming for now though that length of x is small
        prev_cached = True
        if (self.x == 0).all().item(): self.x = x; prev_cached = False
        else: self.x = torch.cat([self.x, x.unsqueeze(1)], dim = 1)
        b, _, _ = x.shape # now x of shape (b, 1, id)
        # pretty much just calculate the entire KV cache by calculating Q and V
        new_Q = torch.matmul(x, self.W_Q).view(b, 1, self.heads, self.hidden).permute(0, 2, 1, 3)
        new_K = torch.matmul(x, self.W_K).view(b, 1, self.heads, self.hidden).permute(0, 2, 1, 3)
        new_V = torch.matmul(x, self.W_V).view(b, 1, self.heads, self.hidden).permute(0, 2, 1, 3)
        # TODO: I had this problem where I made V be input_dim instead of hidden_dim, need to check if this is fixed overall
        if prev_cached:
            self.K = torch.cat([self.K, new_K], dim = 2)
            self.V = torch.cat([self.V, new_V], dim = 2) # all of shape (b, h, s+1, d)
        else:
            self.K = new_K
            self.V = new_V
        scores = torch.matmul(new_Q, self.K.transpose(-2, -1))
        attn = torch.softmax(scores, dim = -1)
        almost = torch.matmul(attn, self.V).permute(0, 2, 1, 3).contiguous().view(b, 1, self.heads * self.input_dim)
        output = torch.matmul(almost, self.W_O)
        return output

    def backward(self, grad_o: torch.Tensor):
        """Attention backwards is a bit complicated"""
        b, s, _ = grad_o.shape # _ is id
        # first we find grad_W_O, which is easy, since o = almost * W_O, so grad_W_O = almost^T * grad_o
        self.grad_W_O = torch.matmul(self.almost.view(b*s, -1).T, grad_o.view(b*s, -1)) # shape (h*id, id)
        # grad_almost is also easy, since it's grad_o * W_O^T
        grad_almost = torch.matmul(grad_o, self.W_O.T) # shape (b, s, h*d)
        # Next we get grad_V and grad_attn
        grad_almost = grad_almost.view(b, s, self.heads, self.hidden).permute(0, 2, 1, 3) # shape (b, h, s, d)
        grad_V = torch.matmul(self.attn.transpose(-2, -1), grad_almost) # shape (b, h, s, d)
        grad_attn = torch.matmul(grad_almost, self.V.transpose(-2, -1)) # shape (b, h, s, s)
        # Use grad_V to get grad_W_V and grad_x, since V = x * W_V
        grad_V = grad_V.permute(0, 2, 1, 3).contiguous().view(b*s, self.heads * self.hidden) # shape (b*s, h*d)
        self.grad_W_V = torch.matmul(self.x.view(b*s, -1).T, grad_V) # shape (input_dim, h*d)
        grad_x_V = torch.matmul(grad_V, self.W_V.T).view(b, s, self.input_dim)
        # Now we need to get grad_scores from grad_attn
        # Say y = softmax(x) for a 1d vector, then dy/dx = diag(y) - y*y^T, so grad_x = grad_y * (diag(y) - y*y^T)
        # So dL/dx_j = Σ_i dL/dy_i · y_i(δ_ij - y_j) = y_j·g_j - y_j·Σ_i g_i·y_i = y_j · (g_j - Σ_i g_i·y_i) where g_i is upstream gradient
        grad_scores = self.attn * (grad_attn - torch.sum(grad_attn * self.attn, dim = -1, keepdim = True)) # shape (b, h, s, s)
        # Now we can get grad_Q and grad_K from grad_scores, since scores = mask(QK^T)
        # Nice thing is grad scores already 0 on mask
        grad_scores = grad_scores / (self.hidden ** 0.5) # shape (b, h, s, s) Converting to raw scores
        grad_Q = torch.matmul(grad_scores, self.K) # shape (b, h, s, d)
        grad_K = torch.matmul(grad_scores.transpose(-2, -1), self.Q) # shape (b, h, s, d)
        # Now we can get grad_W_Q and grad_W_K from grad_Q and grad_K, since Q = x * W_Q and K = x * W_K
        grad_Q = grad_Q.permute(0, 2, 1, 3).contiguous().view(b*s, self.heads * self.hidden) # shape (b*s, h*d)
        grad_K = grad_K.permute(0, 2, 1, 3).contiguous().view(b*s, self.heads * self.hidden) # shape (b*s, h*d)
        self.grad_W_Q = torch.matmul(self.x.view(b*s, -1).T, grad_Q) # shape (input_dim, h*d)
        self.grad_W_K = torch.matmul(self.x.view(b*s, -1).T, grad_K) # shape (input_dim, h*d)
        grad_x_Q = torch.matmul(grad_Q, self.W_Q.T).view(b, s, self.input_dim)
        grad_x_K = torch.matmul(grad_K, self.W_K.T).view(b, s, self.input_dim)
        # Finally, grad_x is the sum of grad_x_V, grad_x_Q, and grad_x_K, plus the residual connection from grad_o
        self.grad_x = grad_x_V + grad_x_Q + grad_x_K
        return self.grad_x

    def step(self, lr):
        self.W_Q.data -= lr * self.grad_W_Q
        self.W_K.data -= lr * self.grad_W_K
        self.W_V.data -= lr * self.grad_W_V
        self.W_O.data -= lr * self.grad_W_O

    def zero_grad(self):
        self.grad_W_Q = torch.zeros_like(self.W_Q)
        self.grad_W_K = torch.zeros_like(self.W_K)
        self.grad_W_V = torch.zeros_like(self.W_V)
        self.grad_W_O = torch.zeros_like(self.W_O)

    def params(self):
        return [self.W_Q, self.W_K, self.W_V, self.W_O]


class ReLU():
    def forward(self, x: torch.Tensor):
        self.x = x
        return torch.maximum(x, torch.zeros_like(x))

    def backward(self, grad_o: torch.Tensor):
        return grad_o * (self.x > 0)

    def step(self, lr):
        pass

    def zero_grad(self):
        pass

    def params(self):
        return []


class LayerNorm():
    def __init__(self, input_dim, eps = 1e-5):
        self.input_dim = input_dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.b = nn.Parameter(torch.zeros(input_dim))
        # Intermediate vars
        self.x = torch.zeros(1, 1, input_dim)
        self.mean = torch.zeros(1, 1, input_dim)
        self.var = torch.ones(1, 1, input_dim)
        self.x_hat = torch.zeros(1, 1, input_dim)
        # Gradients
        self.grad_gamma = torch.zeros(input_dim)
        self.grad_b = torch.zeros(input_dim)

    def forward(self, x:torch.Tensor):
        self.x = x # shape (b, s, id) 
        self.mean = torch.mean(x, dim = -1, keepdim = True)
        self.var = torch.var(x, dim = -1, keepdim = True, unbiased = False) # Using Bessel's correction is not necessary
        self.x_hat = (x - self.mean) / torch.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.b

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_o: torch.Tensor):
        # Grad wrt gamma and b are easier
        # grad_o is of shape (b, s, id)
        self.grad_b = torch.sum(grad_o, dim = (0, 1)) # shape (id)
        # Since o = gamma * x + b, df/dx_i = \sum_j df/do_j * do_j/dx_i, since o_j = x_j * gamma_j + b_j, do_j = dx_i * gamma_i * delta_ij + db_i * delta_ij
        # So do_j/dx_i = delta_ij * gamma_i, so df/dx_i = df/do_i * gamma_i and df/dgamma_i = df/do_i * x_hat_i
        self.grad_gamma = torch.sum(grad_o * self.x_hat, dim = (0, 1)) # shape (id)
        # Getting back to x
        grad_x_hat = grad_o * self.gamma # shape(b, s, id)
        # As x_hat = (x - mean) / sqrt(var + eps), df/dx_i = df/dx_hat_i * dx_hat_i/dx_i + df/dmean * dmean/dx_i + df/dvar * dvar/dx_i
        # df/dmean = \sum_j df/dx_hat_j * dx_hat_j/dmean, and dx_hat_j/dmean = -1/sqrt(var + eps), so df/dmean = -\sum_j df/dx_hat_j / sqrt(var+eps)
        grad_mean = -torch.sum(grad_x_hat, dim = -1, keepdim = True) / torch.sqrt(self.var + self.eps) # shape (b, s, 1)
        # df/dvar = \sum_j df/dx_hat_j * dx_hat_j/dvar, dx_hat_j/dvar = -0.5 * (x_j - mean) / (var + eps)^(3/2)
        # so df/dvar = -0.5 \sum_j df/dx_hat_j * (x_hat_j) / (var + eps)^{1/2}
        grad_var = -0.5 * torch.sum(grad_x_hat * self.x_hat, dim = -1, keepdim = True) / (self.var + self.eps)
        # dvar / dx_i = 2x_i / input_dim - 2 * mean / input_dim = 2(x_i - mean)/input_dim
        grad_x = grad_x_hat / torch.sqrt(self.var + self.eps) + (grad_mean + 2 * grad_var * (self.x - self.mean)) / self.input_dim
        return grad_x

    def step(self, lr):
        self.gamma.data -= lr * self.grad_gamma
        self.b.data -= lr * self.grad_b

    def zero_grad(self):
        self.grad_gamma = torch.zeros(self.input_dim)
        self.grad_b = torch.zeros(self.input_dim)

    def params(self):
        return [self.gamma, self.b]

# Now onto the actual transformer architecture

class MLP():
    def __init__(self, dim: int, inner_dim = None, layers: int = 1, act = ReLU):
        self.l = layers
        self.out_dim = dim
        if inner_dim == None: inner_dim = 4 * dim # As per GPT architecture inner dim is 4 * outer for the MLP
        self.in_dim = inner_dim
        self.act = act
        if layers < 0:
            raise ValueError("mlp must have non-negative layers")
        elif layers == 0:
            self.layers = [linear(dim)]
        else:
            self.layers = [linear(dim, inner_dim), act()]
            for i in range(layers - 1):
                self.layers.append(linear(inner_dim))
                self.layers.append(act())
            self.layers.append(linear(inner_dim, dim))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_o: torch.Tensor):
        for layer in reversed(self.layers):
            grad_o = layer.backward(grad_o)
        return grad_o

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def params(self):
        return [param for layer in self.layers for param in layer.params()]


class Transformer_Block():
    def __init__(self, dim, attn_hid, heads):
        self.dim = dim
        self.attention_hidden_dim = attn_hid
        self.attn_heads = heads
        self.LN1 = LayerNorm(dim)
        self.attn = Attention(dim, attn_hid, heads)
        self.LN2 = LayerNorm(dim)
        self.MLP = MLP(dim)

    def forward(self, x: torch.Tensor):
        x = x + self.attn.forward(self.LN1.forward(x))
        x = x + self.MLP.forward(self.LN2.forward(x))
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_o: torch.Tensor):
        grad_o = self.MLP.backward(grad_o)
        grad_o = self.LN2.backward(grad_o)
        grad_o = self.attn.backward(grad_o)
        grad_o = self.LN1.backward(grad_o)
        return grad_o

    def step(self, lr):
        self.MLP.step(lr)
        self.LN2.step(lr)
        self.attn.step(lr)
        self.LN1.step(lr)

    def zero_grad(self):
        self.MLP.zero_grad()
        self.LN2.zero_grad()
        self.attn.zero_grad()
        self.LN1.zero_grad()

    def params(self):
        return self.LN1.params() + self.attn.params() + self.LN2.params() + self.MLP.params()

# Token and Position Embed

class Embed():
    # Basically a linear layer with
    def __init__(self, hidden_dim: int, vocab_size: int, pos_embed = "cosine", max_s = 30000):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.pos_embed_type = pos_embed
        self.max_seq_len = max_s
        self.tok_embed = nn.Parameter(torch.normal(0, (1/hidden_dim) ** 0.5, size = (vocab_size, hidden_dim)))
        if (pos_embed == "cosine"):
            pos = torch.arange(max_s).unsqueeze(1)
            i = torch.arange(hidden_dim)
            angle_rates = 1 / (10000 ** (2 * (i // 2) / hidden_dim)) # 10000 is the standard, the pos embed cycles every 10000
            angles = pos * angle_rates
            pe = torch.zeros(max_s, hidden_dim)
            pe[:, 0::2] = torch.sin(angles[:, 0::2])
            pe[:, 1::2] = torch.cos(angles[:, 1::2])
            self.pos_embed = pe
        elif (pos_embed == "random"):
            self.pos_embed = nn.Parameter(torch.normal(0, (1/hidden_dim) ** 0.5, size = (max_s, hidden_dim)))
            # Gradient
            self.grad_pos_embed = torch.zeros(max_s, hidden_dim)
        # For backwards computation later
        self.x = torch.zeros(1, 1)
        self.grad_tok_embed = torch.zeros(vocab_size, hidden_dim)

    def forward(self, x: torch.Tensor):
        # x of shape (b, s)
        self.x = x
        b, s = x.shape
        return self.tok_embed[x] + self.pos_embed[:s].unsqueeze(0)

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def backward(self, grad_o: torch.Tensor):
        b, s, _ = grad_o.shape
        self.grad_tok_embed = torch.zeros_like(self.tok_embed)
        self.grad_tok_embed.index_add_(0, self.x.reshape(-1), grad_o.reshape(b * s, -1)) # 0 says we are indexing along the 0 dimension
        if (self.pos_embed_type == "random"):
            self.grad_pos_embed = torch.zeros_like(self.pos_embed)
            self.grad_pos_embed[:s] += grad_o.sum(dim = (0))
        return None

    def step(self, lr):
        self.tok_embed.data -= lr * self.grad_tok_embed
        if (self.pos_embed_type == "random"):
            self.pos_embed.data -= lr * self.grad_pos_embed

    def zero_grad(self):
        self.grad_tok_embed = torch.zeros_like(self.grad_tok_embed)
        if (self.pos_embed_type == "random"):
            self.grad_pos_embed = torch.zeros_like(self.grad_pos_embed)

    def params(self):
        return [self.tok_embed, self.pos_embed] if self.pos_embed_type == "random" else [self.tok_embed]

# LLM
# I will just import a tokenizer

class LLM():
    def __init__(self, layers: int, hid_dim: int, attn_dim: int, heads: int, vocab_size: int, pos_embed = "cosine", max_s = 30000):
        self.name = "Lorial v0"
        self.meta_data = {
            "layers": layers,
            "hidden dimension": hid_dim,
            "attention dimension": attn_dim,
            "attention heads": heads, 
            "position embedding": pos_embed, 
            "vocab size": vocab_size, 
            "maxmium sequence length": max_s
        }
        self.embed = Embed(hid_dim, vocab_size, pos_embed, max_s)
        self.layers = [Transformer_Block(hid_dim, attn_dim, heads) for _ in range(layers)]
        self.ln_f = linear(hid_dim, vocab_size)

    def forward(self, x):
        o = self.embed(x)
        for block in self.blocks:
            o = block.forward(o)
        o = self.ln_f.forward(o)
        return self.head.forward(o)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_o):
        grad_o = self.ln_f.backward(grad_o)
        for layer in self.layers:
            grad_o = layer.backward(grad_o)
        grad_o = self.embed.backward(grad_o)
        return grad_o

    def step(self, lr):
        self.embed.step(lr)
        for layer in self.layers:
            layer.step(lr)
        self.ln_f.step(lr)

    def zero_grad(self):
        self.embed.zero_grad()
        for layer in self.layers:
            layer.zero_grad()
        self.ln_f.zero_grad()

    # Generate sentences
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, topk: int = None):
        # idx is (b, s) or just s
        if (len(idx.shape) == 1):
            idx.unsqueeze(0)
        for _ in range(max_new_tokens):
            s = idx.shape[1]
            idx_clip = idx if s < self.embed.max_seq_len else idx[:, -self.embed.max_seq_len:]
            logits = self.forward(idx_clip) # (b, s, vocab)
            logits = logits[:, -1, :] # (b, 1, vocab) last tok
            logits = logits / temperature
            if topk is not None:
                v, _ = torch.topk(logits, topk)
                logits[logits < v[:, -1]] = float('-inf')
            prob = torch.softmax(logits, dim = -1)
            next_id = torch.multinomial(prob, num_samples = 1) # shape (b, 1)
            idx = torch.cat([idx, next_id], dim = 1) # shape (b, s+l) where l is the no. loops
        return idx

    # TODO: KV_cache generation

    # General utils
    def params(self):
        return self.embed.params() + self.ln_f.params() + [p for layer in self.layers for p in layer.params()]

    def param_count(self):
        return sum(p.numel() for p in self.params())

    def print(self):
        print(self.name + " model specs")
        print("----------Metadata----------")
        print(self.meta_data)
        print("----------Param Count----------")
        print(f"Total params: {self.param_count():,}")
