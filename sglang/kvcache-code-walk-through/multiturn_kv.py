#!/usr/bin/env python3
import torch
from graphviz import Digraph
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids = node.key
    text = tok.decode(ids, skip_special_tokens=True) if tok and ids else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth + 1, tok)

def graphviz_dump(root: TreeNode, out_path="kv_tree.dot"):
    dot = Digraph(format="png")
    def visit(node, uid="root"):
        label = "".join("█" for _ in node.key) or "·"
        dot.node(uid, f"{label}\\n(len={len(node.key)})")
        for i, c in enumerate(node.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(c, cid)
    visit(root)
    dot.save(out_path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_fast=False,
        use_auth_token=True
    )
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        load_in_8bit=False,
        use_auth_token=True
    )
    cache = RadixCache(None, None, page_size=1, disable=False)
    essay = (
        "Once upon a time in a galaxy far away, there lived an explorer who "
        "sought knowledge beyond the stars. The explorer built a ship with "
        "advanced technology and embarked on a journey to discover new worlds."
    )
    prompts = [
        "Refine grammar and style of this essay",
        "The essay still needs refinement, please convert all verbs to past tense",
        "The essay still needs refinement, please tighten the introduction",
        "The essay still needs refinement, please shorten the conclusion",
        "The essay still needs refinement, please improve overall flow and coherence"
    ]
    history_ids = []
    for instr in prompts:
        prompt_text = essay + "\n\n" + instr
        new_ids = tok.encode(prompt_text, add_special_tokens=False)
        all_ids = history_ids + new_ids
        cache.insert(all_ids)
        inputs = torch.tensor([all_ids], device=device)
        cfg = GenerationConfig(max_new_tokens= len(new_ids) + 200, do_sample=False, pad_token_id=tok.pad_token_id)
        out = model.generate(inputs, generation_config=cfg)[0].tolist()
        resp_ids = out[len(all_ids):]
        essay = tok.decode(resp_ids, skip_special_tokens=True).strip()
        cache.insert(resp_ids)
        history_ids = all_ids + resp_ids
        print("\nCurrent KV-cache tree:")
        dump(cache.root_node, tok=tok)
    graphviz_dump(cache.root_node)
    print("Wrote kv_tree.dot")
if __name__ == "__main__":
    main()




