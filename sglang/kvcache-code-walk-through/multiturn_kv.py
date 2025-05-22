#!/usr/bin/env python3
import sys
# 1) Make sure we pick up your local clone first:
sys.path.insert(0, "/sgl-workspace/sglang/python")

from transformers import GPT2Tokenizer
from sglang.srt.mem_cache.radix_cache import RadixCache

def dump(node, depth=0):
    indent = "  " * depth
    # try to decode bytes to a printable prefix
    try:
        label = node.prefix.decode("utf-8")
    except:
        label = repr(node.prefix)
    print(f"{indent!r:<12}  (len={len(node.prefix):2}, refs={node.ref_count})  → {label!r}")
    for child in getattr(node, "children", []):
        dump(child, depth + 1)

def main():
    # 1) load GPT-2 tokenizer so we get real token-IDs
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    # 2) make an empty radix-tree
    cache = RadixCache()

    # 3) multi-turn prompts
    prompts = [
        "Hello, how are you?",
        "Can you tell me a joke?",
        "Why is that funny?"
    ]

    for turn, text in enumerate(prompts, start=1):
        ids = tok.encode(text, add_special_tokens=False)
        # insert every prefix of the token-ID sequence
        for L in range(1, len(ids) + 1):
            prefix_bytes = bytes(ids[:L])
            cache.insert(prefix_bytes)

        print(f"\n=== After turn {turn} (‘{text}’) ===")
        dump(cache._root)

if __name__ == "__main__":
    main()
