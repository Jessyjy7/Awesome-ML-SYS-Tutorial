#!/usr/bin/env python3
import sys
sys.path.insert(0, "/sgl-workspace/sglang/python")

from transformers import GPT2Tokenizer
from sglang.srt.mem_cache.radix_cache import RadixCache

def dump(node, depth=0):
    indent = "  " * depth
    try:
        label = node.prefix.decode("utf-8")
    except:
        label = repr(node.prefix)
    print(f"{indent!r:<12}  (len={len(node.prefix):2}, refs={node.lock_ref})  → {label!r}")
    for child in node.children.values():
        dump(child, depth + 1)

def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # ───────────── CORRECTION HERE ─────────────
    # Pass None for the two pools, and page_size=1
    cache = RadixCache(
        None,                     # req_to_token_pool
        None,                     # token_to_kv_pool_allocator
        page_size=1,              # smallest granularity
        disable=False             # keep the tree enabled
    )  ## <<<<<<
    # ────────────────────────────────────────────

    prompts = [
        "Hello, how are you?",
        "Can you tell me a joke?",
        "Why is that funny?",
    ]

    for turn, text in enumerate(prompts, start=1):
        ids = tok.encode(text, add_special_tokens=False)

        # insert the *full* sequence as one key
        # (or, if you want every prefix: loop L in 1..len(ids))
        cache.insert(ids)

        print(f"\n=== After turn {turn} (‘{text}’) ===")
        dump(cache.root_node)

if __name__ == "__main__":
    main()
