#!/usr/bin/env python3
import sys
from graphviz import Digraph
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

# ———————————————— Text dump helper ————————————————
def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids = node.key
    text = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for c in node.children.values():
        dump(c, depth+1, tok)

# ———————————————— Graph dump helper ————————————————
def graphviz_dump(root: TreeNode, out_path="kv_tree.dot"):
    dot = Digraph("kv_cache", format="png")
    def visit(node: TreeNode, uid="root"):
        label = "".join(chr(0x2588) for _ in node.key) or "·"
        # use a block‐count to hint at length, you can tweak
        dot.node(uid, f"{label}\\n(len={len(node.key)})")
        for i, child in enumerate(node.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(child, cid)
    visit(root)
    dot.save(out_path)
    print(f"\nGraphviz tree written to {out_path!r}.")
    print("   Render with: dot -Tpng kv_tree.dot -o kv_tree.png")

# —————————————————————————————————————————————————————————————————————————————————————————————
def main():
    # 1) Load model & tokenizer
    tok   = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium").eval()

    # 2) Empty RadixCache
    cache = RadixCache(None, None, page_size=1, disable=False)

    # 3) Multi-turn prompts with both shared and distinct prefixes
    prompts = [
      "Hello, how are you?",
      "Hello, how is your day going?",
      "What's the weather like in Paris today?",
      "Hello, how are you now after our talk?"
    ]

    for turn, text in enumerate(prompts, 1):
        print(f"\n=== Turn {turn}: Prompt → {text!r} ===")
        ids = tok.encode(text, add_special_tokens=False)
        cache.insert(ids)
        print("Prompt inserted.")

        inputs = tok(text, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_length=inputs.input_ids.shape[1] + 80,
            do_sample=True, top_k=50, temperature=0.7,
            pad_token_id=tok.pad_token_id
        )[0].tolist()
        resp_ids = out[len(ids):]
        reply = tok.decode(resp_ids, skip_special_tokens=True)
        print(f"GPT-2 replies → {reply!r}")

        cache.insert(resp_ids)
        print("Response inserted.")

        print("\nText dump of KV-cache:")
        dump(cache.root_node, tok=tok)

    # 4) At the end, also emit a Graphviz .dot
    graphviz_dump(cache.root_node, out_path="kv_tree.dot")

if __name__ == "__main__":
    main()
