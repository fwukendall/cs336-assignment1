# CS336 Assignment 1 Write-up

JC Chen

## Unicode 1

**a. What Unicode character does chr(0) return?**

A null character, or "\x00"


**b. How does this character’s string representation (__repr__()) differ from its printed representation?**

Its string repr is "\x00" but when you print it, it's empty.

**c.  What happens when this character occurs in text?**

It doesn't show up in text when you print it.

## Unicode 2

**a. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.**

Training corpus are mostly ASCII characters since they'll be predominantly English or at least english-character-based. UTF-16/32 will be much less space efficient on that front since they use more bytes for the most common characters, and only more efficient than UTF-8 under specific languages like Chinese or Korean.


**b. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.**

```
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Not all UTF-8 characters are single-byte. This function breaks down the byte-string into single byte and then transform, when in fact some of the original characters could be 2-bytes or more.

For example, the japanese hello b'\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf' will not decode under this function.


**c. Give a two byte sequence that does not decode to any Unicode character(s).**

```
broken = False
for i in range(256):
    try:
        c = bytes([i]).decode('utf-8')
        continue
    except:
        print(i)
    for j in range(256):
        try:
            c = bytes([i, j]).decode('utf-8')

            continue
        except:
            broken = True
            break
    if broken:
        break 

print((i, j))
``` 

The first such 2-bytes that it found is (128, 0), aka b'\x80\x00'



## Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

**(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection.**

**How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?**

Training took 4m25s on the TinyStories dataset (16 processes in pre-tokenization). Peak memory usage is 2.2GB (in pre-tokenization).
The longest token is b' responsibility'. Makes sense! For children's stories, that's probably the longest high-frequency word.


**(b) Profile your code. What part of the tokenizer training process takes the most time?**

Total runtime is 4m25s.
Pre-tokenization took roughly 55 seconds, And then on average every 500 merges takes 22 seconds.
The merging operation takes the longest amount of time: with the biggest culprit being the in-word update and token-pair count sort.


## Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

**(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection.**

**What is the longest token in the vocabulary? Does it make sense?**

The longest token is `'-' * 64` (64 dashes). It probably makes sense as a webpage separator.

**(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.**

Both are English only, yet the OpenWebText tokens have a lot more non-words like dashes and equal-signs. Average length of token wise, there's not much of a difference, when I compare the top 10000 in the vocab their average length turns out to be quite similar.

## Problem (tokenizer_experiments): Experiments with tokenizers (4 points)

**(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?**

Deliverable: A one-to-two sentence response.


I made 100 seeded samples of 10-doc lists from TinyStories.  Average length of TinyStory in bytes is 811.197 bytes, and encoded token length is 198.9 tokens, so a compression ratio of 4.08

I made 50 seeded samples of 10-docs lists from OWT. Average length in bytes is 4963 bytes, average encoded token length is 1129 tokens, with a c-r of 4.39

This compression isn't super fair because with a larger vocab, the owt encoder probably will use more space per token.


**(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.**

Deliverable: A one-to-two sentence response.

It takes half the amount of time, but end up with 37% more tokens (compression ratio changes from 4.39 to 3.26)


**(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?**
Deliverable: A one-to-two sentence response.

OWT takes about 52.9 secs to process 2.48MB of text, a throughput of 47KB/sec. Without parallelization 4888 hours to tokenize the Pile dataset, or 203 days.

**(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?**

Deliverable: A one-to-two sentence response.

The OWT bpe vocab size is 32000, under the max-int under uint16 which is 65535. Therefore this is the tightest choice (of integer-byte dtypes) we may use.

## Problem (transformer_accounting): Transformer LM resource accounting (5 points)
**(a) Consider GPT-2 XL, which has the following configuration:**

```
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
```

**Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?**

Deliverable: A one-to-two sentence response.

- Input and Output embedding each has 80411200 params;
- Each layer has 40963200 params;
- Final RMSNorm has 1600 params;
- Total params is then 2,127,057,600;
- Using float32 to store params would result in roughly 8GB of memory required.

**(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.  Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.**

matmul happens at:
- each layer of transformer:
    - attention:
        - qkv projection: 3 * 2 * 1024 * 1600**2 = 15.73GFLOPs
        - qk mult: 25 * 2 * 1024 * 1024 * 64 = 3.36GFLOPs
        - softmax-v mult: 25 * 2 * 1024 * 1024 * 64 = 3.36GFLOPs
        - out projtion: 2 * 1024 * 1600**2 = 5.24GFLOPs
    - ffn:
        - 2 * 3 * 1024 * 6400 * 1600 = 62.91GFLOPs
    - total is 90.59 GFLOPs per layer
- output embedding: 1024 * 2 * 1600 * 50257 = 164.68GFLOPs

So total would be around 90.59 * 48 + 164.68 = 4.513TFLOPs

**(c) Based on your analysis above, which parts of the model require the most FLOPs?**

Deliverable: A one-to-two sentence response.

It is the FFNs which has the largest matrix multiplication (with 3 of them, in SwiGLU)
Second place would be the qkv and output projections in the attention.


**(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?**

Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

(assuming d_ff = 4 * d_model)

```
def calc_tflops(seq_len, d_model, num_layers):
    attn_pl = 3 * 2 * seq_len * d_model * d_model + \
    2 * seq_len * seq_len * d_model + \
    2 * seq_len * d_model * d_model + \
    2 * seq_len * d_model * d_model
    attn_total = attn_pl * num_layers

    ffn_pl = 3 * 2 * seq_len * d_model * d_model * 4
    ffn_total = ffn_pl * num_layers

    out_total = 2 * seq_len * d_model * 50273

    res = np.round(np.array([
        [attn_pl, attn_total],
        [ffn_pl, ffn_total],
        [out_total, out_total]
    ]) / 1e9, 2)
    res2 = res[:, 1:] / res[:, 1:].sum()
    return np.round(np.concat([res, res2], axis=1), 3)
```

GPT-2 small

- attention: 8.05GFLOPs/layer * 12 = 96.64GFLOPs (27.6%)
- FFN : 14.49GFLOPs/layer * 12 = 173.95GFLOPs (49.7%)
- outpus: 79.07GFLOPs (22.9%)

GPT-2 medium

- attention: 12.88GFLOPs/layer * 24 = 309.24GFLOPs (29.9%)
- FFN : 25.76GFLOPs/layer * 24 = 618.24GFLOPs (59.9%)
- outpus: 105.43GFLOPs (10.2%)

GPT-2 large

- attention: 18.79GFLOPs/layer * 36 = 676.46GFLOPs (30.0%)
- FFN : 40.26GFLOPs/layer * 36 = 1449.36GFLOPs (64.2%)
- outpus: 164.73GFLOPs (5.8%)

As the model scales up, the output layer matmul occupies a smaller pct of compute, and of the remaining two big computes, FFN occupies a faster-growing pct than attention. This is due to the fact that the output layer scales linearly to d_model, and attention-per-layer scales `in-part quadratic, in-part linear` to d_model, and FFN-per-layer scales quadraticlly to d_model, while the total latter two also scales linearly to num_layers.


**(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?**

Deliverable: A one-to-two sentence response.

The attention-per-layer part scales `in-part quadratic` to context_length, so that will increase its percentage in total flops. An accounting is as follow:

- attention: 2053.53GFLOPs/layer * 48 = 98.57TFLOPs (65.9%)
- FFN : 1006.63GFLOPs/layer * 48 = 48.32TFLOPs (32.3%)
- outpus: 2.636TFLOPs (1.8%)

## Problem (learning_rate_tuning): Tuning the learning rate (1 point)
**As we will see, one of the hyperparameters that affects training the most is the learning rate. Let’s see that in practice in our toy example. Run the SGD example above with three other values for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over the course of training)?**

Deliverable: A one-two sentence response with the behaviors you observed

Theoretically loss should approach 0. With a manual_seed, my initial loss was 28.0;
- lr = 1.0: loss comes down slowly and ends at 23.2;
- lr = 1e1: loss comes down faster and ends at 3.76;
- lr = 1e2: loss comes down rapidly and ends at 4e-23;
- lr = 1e3: loss blows up from the get-go, ends up at 3e18:


## Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

**Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.**


**(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 4 × d_model.**

For simplicity, when calculating memory usage of activations, consider only the following components:

```
- Transformer block
- RMSNorm(s)
– Multi-head self-attention sublayer: QKV projections, Q⊤K matrix multiply, softmax, weighted sum of values, output projection.
– Position-wise feed-forward: W1 matrix multiply, SiLU, W2 matrix multiply
- final RMSNorm
- output embedding
- cross-entropy on logits
```

Deliverable: An algebraic expression for each of parameters, activations, gradients, and optimizer state, as well as the total.

num_params = d_model * (num_layers * (
    2 # norm1 and norm2
    + 3 * d_model # qkv
    + d_model # out
    + 2 * 4 * d_model # ffn
) + 1 # final norm
+ vocab_size # in embed
+ vocab_size # out embed
) = d_model * (num_layers * (12 * d_model + 2) + 1 + 2 * vocab_size)

num_adamstate = 2 * num_params

num_grad = num_params

num_activations = batch_size * context_length * (
    d_model # in embed
    + num_layers * (
        2 * d_model # norm1 and norm2
        + 3 * d_model # qkv proj
        + context_length * num_heads # qtk
        + context_length * num_heads # softmax
        + d_model # val wsum
        + d_model # out proj
        + 2 * 4 * d_model # ffn w1 + w2 + silu
    )
    + d_model # final norm
    + vocab_size # out embed
) = batch_size * context_length * (2 * d_model + vocab_size
    + num_layers * (19 * d_model + 2 * context_length * num_heads))

total_mem_use = dtype_num_bytes * (4 * num_params + num_activations)


**(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?**

Deliverable: An expression that looks like a · batch_size + b for numerical values a, b, and a number representing the maximum batch size.

mem_use_bytes = 4 * 4 * num_params + 4 * num_activations
For GPT-2 XL:
mem_use_bytes = 26_168_601_600 + 15_318_454_272 * batch_size

80GB memory = 85_899_345_920 bytes
max_batch_size = 3

**(c) How many FLOPs does running one step of AdamW take?**

Deliverable: An algebraic expression, with a brief justification.

For one forward step, I use the following formula

```
def calc_tflops(seq_len, d_model, num_layers):
    attn_pl = 3 * 2 * seq_len * d_model * d_model + \
    2 * seq_len * seq_len * d_model + \
    2 * seq_len * d_model * d_model + \
    2 * seq_len * d_model * d_model
    attn_total = attn_pl * num_layers

    ffn_pl = 3 * 2 * seq_len * d_model * d_model * 4
    ffn_total = ffn_pl * num_layers

    out_total = 2 * seq_len * d_model * 50273

    res = np.round(np.array([
        [attn_pl, attn_total],
        [ffn_pl, ffn_total],
        [out_total, out_total]
    ]) / 1e9, 2)
    res2 = res[:, 1:] / res[:, 1:].sum()
    return np.round(np.concat([res, res2], axis=1), 3)
```

matmul_forward_flops = batch_size * seq_len * d_model * (num_layers * (26 * d_model + 2 * seq_len) + 2 * vocab_size)

For grad computation and adamw computation, FLOPs would be:

num_flops_grad ~= matmul_forward_flops * 2 # derivative, then prop
num_flops_adamw = num_params * 15 # see annotated code below

```
# python

grad = p.grad.data 
m = b1 * m + (1 - b1) * grad      # 2 * n_p
v = b2 * v + (1 - b2) * grad**2   # + 3 * n_p
lr_t = lr * np.sqrt(1 - b2**t) / (1 - b1**t) # + 3 * n_p
p.data -= lr_t * m / (torch.sqrt(v) + eps) # + 4 * n_p
p.data -= lr * wd * p.data # + 3 * n_p
```

recall that num_params ~= d_model * (num_layers * (12 * d_model + 2) + 1 + 2 * vocab_size); with seq_len roughly equal to d_model, and vocab_size roughly equal to `num_layers * d_model` (strong assumption! but roughly right for GPT2-XL), we can shorten the formulas

```
num_params ~= 14 * num_layers * d_model ** 2
matmul_forward_flops_pertoken ~= 30 * num_layers * d_model ** 2
matmul_forward_flops ~= 2.15 * seq_len * num_params * batch_size
num_flops_grad ~= 4.3 * seq_len * num_params * batch_size
```

Notice that the amount of compute inside optimizer (excluding grad()) is negligible since 6.3 * seq_len >> 15. So the compute is roughly 6.3 * num_tokens * num_params


**(d) model flops utilization (mfu) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak flop throughput [chowdhery et al., 2022]. an nvidia a100 gpu has a theoretical peak of 19.5 teraflop/s for float32 operations. assuming you are able to get 50% mfu, how long would it take to train a gpt-2 xL for 400K steps and a batch size of 1024 on a single A100? Following Kaplan et al. [2020] and Hoffmann et al. [2022], assume that the backward pass has twice the FLOPs of the forward pass.**

Deliverable: The number of days training would take, with a brief justification

4786 days

num_params ~= 1.6G
num_tokens_perbatch = 1M
flops_per_step = 6.3 * 1.6G * 1M = 10080T
effective_use = 19.5T * 0.5 = 9.75T
time_per_step = 10080FLOPs / 9.75(FLOP/s) = 1034s
time_total = 984s * 400K ~= 4786days
