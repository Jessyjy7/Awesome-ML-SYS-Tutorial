#!/usr/bin/env python3
import sys
# 1) Ensure we load your local SGLang clone first
sys.path.insert(0, "/sgl-workspace/sglang/python")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

def dump(node: TreeNode, depth=0, tok=None):
    """Recursively print each node’s key as decoded text, plus metadata."""
    indent = "  " * depth
    ids = node.key  # List[int]
    # Decode back to text, skipping special tokens
    text = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth + 1, tok)

def main():
    # — Load a larger GPT-2 for more coherent replies
    tok   = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").eval()

    # — Instantiate an “empty” RadixCache (pools=None for simplicity)
    cache = RadixCache(None, None, page_size=1, disable=False)

    # — Four turns: two share “Hello, how…”, one is distinct, one returns to it
    prompts = [
        "Hello, how are you?",
        "Hello, how is your day going?",
        "What's the weather like in Paris today?",
        "Hello, how are you now after our talk?"
    ]

    for turn, text in enumerate(prompts, start=1):
        print(f"\n=== Turn {turn}: Prompt → {text!r} ===")

        # 1) Insert the prompt into the KV-cache
        ids = tok.encode(text, add_special_tokens=False)
        cache.insert(ids)
        print("📨  Prompt inserted into KV-cache.")

        # 2) Generate a reply with extra room for completion
        inputs = tok(text, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 80,   # allow 80 new tokens
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tok.pad_token_id
        )
        out_ids = out[0].tolist()
        resp_ids = out_ids[len(ids):]
        response = tok.decode(resp_ids, skip_special_tokens=True)
        print(f"GPT-2 replies → {response!r}")

        # 3) Insert the response into the KV-cache
        cache.insert(resp_ids)
        print("Response inserted into KV-cache.")

        # 4) Dump the full human-readable Radix tree
        print("\nKV-cache Radix tree:")
        dump(cache.root_node, tok=tok)

if __name__ == "__main__":
    main()
