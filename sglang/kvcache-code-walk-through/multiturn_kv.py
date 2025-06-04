#!/usr/bin/env python3
import sys
import torch
from graphviz import Digraph
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids = node.key
    text = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth + 1, tok)

def graphviz_dump(root: TreeNode, out_path="kv_tree.dot"):
    dot = Digraph("kv_cache", format="png")
    def visit(node: TreeNode, uid="root"):
        label = "".join(chr(0x2588) for _ in node.key) or "·"
        dot.node(uid, f"{label}\\n(len={len(node.key)})")
        for i, child in enumerate(node.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(child, cid)
    visit(root)
    dot.save(out_path)
    print(f"Graphviz tree written to {out_path!r}.")
    print("You can render it with: dot -Tpng kv_tree.dot -o kv_tree.png")

def main():
    # Detect device (cuda if available, else cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Llama 2 7B-Chat tokenizer + model
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=False)
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=False
    )

    # Construct RadixCache with a larger page_size (4)
    cache = RadixCache(None, None, page_size=4, disable=False)

    prompts = [
        "Hello, I have a statement that needs refinement, please change all the I to we in the following statement and keep the rest the same. I start working on SGLang two months ago, I first try to explore the tree structure then visualize it.",
        "Hello, the statement still needs refinement, please change all the verbs to past tense in the statement you just edited and keep the rest the same.",
        "Hello, how is your day going?",
        "What's the weather like in Paris today?",
        "Hello, how are you now after our talk?"
    ]

    for turn, text in enumerate(prompts, start=1):
        print(f"\n=== Turn {turn}: Prompt → {text!r} ===")

        # Insert the prompt into KV-cache
        ids = tok.encode(text, add_special_tokens=False)
        cache.insert(ids)
        print("Prompt inserted into KV-cache.")

        # Generate a reply via Llama 2
        inputs = tok(text, return_tensors="pt").to(device)
        gen_config = GenerationConfig(
            max_new_tokens=200,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tok.pad_token_id
        )
        out = model.generate(**inputs, generation_config=gen_config)
        out_ids = out[0].tolist()
        resp_ids = out_ids[len(inputs.input_ids[0]):]
        reply = tok.decode(resp_ids, skip_special_tokens=True).strip()
        print(f"Llama-2 replies → {reply!r}")

        # Insert the response into KV-cache
        cache.insert(resp_ids)
        print("Response inserted into KV-cache.")

        # Print a human-readable dump of the current Radix tree
        print("\nText dump of KV-cache:")
        dump(cache.root_node, tok=tok)

    # At the end, write out a Graphviz .dot file
    graphviz_dump(cache.root_node, out_path="kv_tree.dot")

if __name__ == "__main__":
    main()
