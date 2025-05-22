#!/usr/bin/env python3
import sys
# 1) Make sure Python picks up your local clone’s “python/” folder first
sys.path.insert(0, "/sgl-workspace/sglang/python")

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids = node.key  # list of token-IDs
    # decode back to text, or show empty string for root
    text = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12}  → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth + 1, tok)

def main():
    # — Load tokenizer & model
    tok   = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # for generation safety
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    # — Instantiate an “empty” radix cache
    cache = RadixCache(None, None, page_size=1, disable=False)

    # — A 4-turn convo with both shared prefixes and a standalone prompt
    prompts = [
        "Hello, how are you?",
        "Hello, how is your day going?",
        "What's the weather like in Paris today?",
        "Hello, how are you now after our talk?"
    ]

    for turn, text in enumerate(prompts, start=1):
        print(f"\n=== Turn {turn}: Prompt: {text!r} ===")

        # 1) Prefill: insert the prompt’s token IDs
        ids = tok.encode(text, add_special_tokens=False)
        cache.insert(ids)
        print("📨  Prompt inserted into KV-cache.")

        # 2) Generate a GPT-2 reply
        inputs = tok(text, return_tensors="pt")
        out_ids = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 20,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tok.pad_token_id
        )[0].tolist()
        resp_ids = out_ids[len(ids):]
        response = tok.decode(resp_ids, skip_special_tokens=True)
        print(f"🤖  GPT-2 replies: {response!r}")

        # 3) Decode: insert the response’s token IDs
        cache.insert(resp_ids)
        print("✅  Response inserted into KV-cache.")

        # 4) Dump the full KV-cache Radix tree
        print("\n🗺️   KV-cache tree:")
        dump(cache.root_node, tok=tok)

if __name__ == "__main__":
    main()
