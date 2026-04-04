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

# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
Deliverable: A one-to-two sentence response.
(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
Deliverable: A one-to-two sentence response.
(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to
tokenize the Pile dataset (825GB of text)?
Deliverable: A one-to-two sentence response.
(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language
model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is
uint16 an appropriate choice?
12
Deliverable: A one-to-two sentence response.