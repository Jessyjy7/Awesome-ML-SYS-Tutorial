#!/usr/bin/env python3
import sys
sys.path.insert(0, "/sgl-workspace/sglang/python")

from transformers import GPT2Tokenizer
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

def dump(node: TreeNode, depth=0):
    indent = "  " * depth
    # node.key is a List[int] for tokens, so decode from bytes if possible
    try:
        # if key was inserted as a bytes object, decode:
        label = bytes(node.key).decode("utf-8")
    except Exception:
        label = repr(node.key)
    print(f"{indent!r:<12}  (len={len(node.key):2}, refs={node.lock_ref})  → {label!r}")
    for child in node.children.values():
        dump(child, depth + 1)

def main():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    cache = RadixCache(None, None, page_size=1, disable=False)

    prompts = [
        "Hello, how are you?",
        "Can you tell me a joke?",
        "Why is that funny?",
    ]

    for turn, text in enumerate(prompts, start=1):
        ids = tok.encode(text, add_special_tokens=False)

        # Insert the full token-ID sequence as a key
        cache.insert(ids)

        print(f"\n=== After turn {turn} (‘{text}’) ===")
        dump(cache.root_node)

if __name__ == "__main__":
    main()
