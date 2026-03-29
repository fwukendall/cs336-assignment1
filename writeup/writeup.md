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

