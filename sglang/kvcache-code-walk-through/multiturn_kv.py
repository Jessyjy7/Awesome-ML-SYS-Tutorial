#!/usr/bin/env python3
import torch
from graphviz import Digraph
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
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
    print(f"Graphviz tree written to {out_path!r}")
    print("Render: dot -Tpng kv_tree.dot -o kv_tree.png")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_fast=False,
        use_auth_token=True
    )
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        device_map="auto" if device=="cuda" else None,
        load_in_8bit=False,
        use_auth_token=True
    )

    cache = RadixCache(None, None, page_size=1, disable=False)

    essay = (
        "Once upon a time in a galaxy far away, there lived an explorer who "
        "sought knowledge beyond the stars. The explorer built a ship with "
        "advanced technology and embarked on a journey to discover new worlds."
    )
    steps = [
        "Refine grammar and style",
        "Convert all verbs to past tense",
        "Tighten the introduction",
        "Shorten the conclusion",
        "Improve overall flow and coherence"
    ]

    history_ids = []
    for i, instruction in enumerate(steps, start=1):
        prompt = f"{instruction}: {essay}"
        print(f"\n=== Turn {i}: {instruction} ===")
        new_ids = tok.encode(prompt, add_special_tokens=False)
        input_ids = history_ids + new_ids

        cache.insert(input_ids)
        print("Inserted full text so far into KV-cache.")

        inputs = torch.tensor([input_ids], device=device)
        gen_cfg = GenerationConfig(
            max_new_tokens=300,
            do_sample=True,
            top_k=50,
            temperature=0.7,
            pad_token_id=tok.pad_token_id
        )
        out = model.generate(inputs, generation_config=gen_cfg)[0].tolist()

        resp_ids = out[len(input_ids):]
        essay = tok.decode(resp_ids, skip_special_tokens=True).strip()
        print(f"LLM output → {essay!r}")

        cache.insert(resp_ids)
        print("Inserted edited essay into KV-cache.")

        history_ids = input_ids + resp_ids

        print("\nCurrent KV-cache tree:")
        dump(cache.root_node, tok=tok)

    graphviz_dump(cache.root_node, out_path="kv_tree.dot")

if __name__ == "__main__":
    main()


