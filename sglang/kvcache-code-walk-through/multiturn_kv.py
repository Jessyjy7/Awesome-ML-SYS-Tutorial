#!/usr/bin/env python3
from transformers import GPT2Tokenizer
from sglang.srt.mem_cache.kv_cache_scheduler import Scheduler, Request
from sglang.srt.mem_cache import tree_cache

# ————————————————————————————————————————————
# Helper to pretty-print the radix tree
def dump(node, depth=0):
    indent = "  " * depth
    prefix = node.prefix.decode(errors="ignore")
    print(f"{indent!r:<12}  (len={len(node.prefix):2}, refs={node.ref_count})")
    for child in getattr(node, "children", []):
        dump(child, depth+1)
# ————————————————————————————————————————————

def main():
    # 1) Load a GPT-2 tokenizer (for real token IDs)
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # 2) Create a scheduler with small cache for demonstration
    sched = Scheduler(max_batch_size=4, max_tokens_cached=64)

    # 3) Define your multi-turn prompts
    prompts = [
        "Hello, how are you?",
        "Can you tell me a joke?",
        "Why is that funny?",
    ]

    for turn, text in enumerate(prompts, start=1):
        # Encode to token IDs (no special tokens)
        ids = tok.encode(text, add_special_tokens=False)

        # Wrap into an SGLang Request
        req = Request(request_id=turn, tokens=ids)

        # ---- “prefill” step: match prefixes & insert new slots ----
        new_reqs = sched.get_new_batch_prefill([req])
        sched.process_batch_result_prefill(new_reqs)

        print(f"\n=== After turn {turn!r} (‘{text}’) ===")
        dump(tree_cache._root)

if __name__ == "__main__":
    main()
