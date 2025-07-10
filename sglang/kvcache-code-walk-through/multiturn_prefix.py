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

def graphviz_dump(root: TreeNode, tok: AutoTokenizer, out_path="kv_tree.dot"):
    dot = Digraph(format="png")
    def visit(node: TreeNode, uid="root"):
        if node.key:
            text = tok.decode(node.key, clean_up_tokenization_spaces=True)
            label = text.replace("\n", "\\n")
        else:
            label = "·"
        dot.node(uid, f"{label}\\n(len={len(node.key)})")
        for i, child in enumerate(node.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(child, cid)
    visit(root)
    dot.save(out_path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", use_fast=False, use_auth_token=True
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
    essay_ids = tok.encode(essay, add_special_tokens=False)
    cache.insert(essay_ids)
    history_ids = essay_ids

    prompts = [
        "Refine grammar and style of this essay",
        "The essay still needs refinement, please convert all verbs to past tense",
        "The essay still needs refinement, please tighten the introduction",
        "The essay still needs refinement, please shorten the conclusion",
        "The essay still needs refinement, please improve overall flow and coherence"
    ]

    for instr in prompts:
        instr_ids = tok.encode(instr, add_special_tokens=False)
        full_prompt = history_ids + instr_ids

        matched_ids, _ = cache.match_prefix(full_prompt)
        matched_len = matched_ids.size(0)
        new_prompt_tokens = full_prompt[matched_len:]
        if new_prompt_tokens:
            cache.insert(new_prompt_tokens)

        inputs = torch.tensor([full_prompt], device=device)
        cfg = GenerationConfig(
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )
        out = model.generate(inputs, generation_config=cfg)[0].tolist()
        resp_ids = out[len(full_prompt):]

        matched_ids, _ = cache.match_prefix(resp_ids)
        matched_len = matched_ids.size(0)
        new_resp_tokens = resp_ids[matched_len:]
        if new_resp_tokens:
            cache.insert(new_resp_tokens)

        history_ids = full_prompt + resp_ids

        print("\nCurrent KV-cache tree:")
        dump(cache.root_node, tok=tok)

    graphviz_dump(cache.root_node, tok, out_path="kv_tree.dot")
    print("Wrote kv_tree.dot")

if __name__ == "__main__":
    main()
