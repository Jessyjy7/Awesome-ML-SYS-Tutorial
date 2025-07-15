#!/usr/bin/env python3
import torch
from graphviz import Digraph
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode

_orig_match_prefix = RadixCache.match_prefix
def _fixed_match_prefix(self, key, **kwargs):
    if self.disable or len(key) == 0:
        return torch.empty((0,), dtype=torch.int64, device=self.device), self.root_node

    if self.page_size != 1:
        page_aligned_len = len(key) // self.page_size * self.page_size
        key = key[:page_aligned_len]

    value_list, last_node = self._match_prefix_helper(self.root_node, key)

    tensor_list = [
        torch.tensor(v, dtype=torch.int64, device=self.device)
        for v in value_list
    ]
    if tensor_list:
        value = torch.cat(tensor_list)
    else:
        value = torch.empty((0,), dtype=torch.int64, device=self.device)

    return value, last_node

RadixCache.match_prefix = _fixed_match_prefix

def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids    = node.key
    text   = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth+1, tok)

def graphviz_dump(root: TreeNode, tok: AutoTokenizer, out_path="kv_tree.dot"):
    dot = Digraph(format="png")
    def visit(node: TreeNode, uid="root"):
        if node.key:
            t     = tok.decode(node.key, clean_up_tokenization_spaces=True)
            label = t.replace("\n","\\n")
        else:
            label = "·"
        dot.node(uid, f"{label}\\n(len={len(node.key)})")
        for i, c in enumerate(node.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(c, cid)
    visit(root)
    dot.save(out_path)

def prefix_length(node: TreeNode) -> int:
    total = 0
    while node.parent:
        total += len(node.key)
        node = node.parent
    return total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        use_fast=False,
        token=True
    )
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        device_map="auto" if device=="cuda" else None,
        load_in_8bit=False,
        token=True
    )

    cache = RadixCache(None, None, page_size=1, disable=False)

    essay = (
        "Once upon a time in a galaxy far away, there lived an explorer who "
        "sought knowledge beyond the stars. The explorer built a ship with "
        "advanced technology and embarked on a journey to discover new worlds."
    )
    history_ids = tok.encode(essay, add_special_tokens=False)
    cache.insert(history_ids)

    prompts = [
        "Refine grammar and style of this essay",
        "The essay still needs refinement, please convert all verbs to past tense",
        "The essay still needs refinement, please tighten the introduction",
        "The essay still needs refinement, please shorten the conclusion",
        "The essay still needs refinement, please improve overall flow and coherence"
    ]

    for instr in prompts:
        instr_ids   = tok.encode(instr, add_special_tokens=False)
        full_prompt = history_ids + instr_ids

        _, last_node = cache.match_prefix(full_prompt)
        mlen         = prefix_length(last_node)
        suffix       = full_prompt[mlen:]
        if suffix:
            cache.insert(suffix)

        inputs = torch.tensor([full_prompt], device=device)
        cfg    = GenerationConfig(max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        out    = model.generate(inputs, generation_config=cfg)[0].tolist()
        resp_ids = out[len(full_prompt):]

        _, last_node = cache.match_prefix(resp_ids)
        mlen         = prefix_length(last_node)
        suffix       = resp_ids[mlen:]
        if suffix:
            cache.insert(suffix)

        history_ids = full_prompt + resp_ids

        print("\nCurrent KV-cache tree:")
        dump(cache.root_node, tok=tok)

    graphviz_dump(cache.root_node, tok, out_path="kv_tree.dot")
    print("Wrote kv_tree.dot")

if __name__ == "__main__":
    main()




