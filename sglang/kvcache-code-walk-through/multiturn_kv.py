import os
import json
import csv
import requests
from transformers import AutoTokenizer
from sglang.srt.mem_cache.radix_cache import RadixCache, TreeNode
from graphviz import Digraph
import matplotlib.pyplot as plt

SG_LANG_PORT    = 50051
MODEL_ID        = "meta-llama/Llama-2-7b"
PROMPTS_FILE    = "/sgl-workspace/MT-Eval/raw_data/refinement_multi_inst.jsonl"
OUT_PRED_JSONL  = "pred_refinement.jsonl"
KV_STATS_CSV    = "kv_stats_refinement.csv"
PLOT_OUT        = "kv_growth.png"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token

def count_nodes(node: TreeNode) -> int:
    return 1 + sum(count_nodes(c) for c in node.children.values())

def max_depth(node: TreeNode) -> int:
    if not node.children:
        return 1
    return 1 + max(max_depth(c) for c in node.children.values())

def dump(node: TreeNode, depth=0, tok=None):
    indent = "  " * depth
    ids = node.key
    text = tok.decode(ids, skip_special_tokens=True) if (tok and ids) else ""
    print(f"{indent!r:<12} → {text!r}  (len={len(ids):2}, refs={node.lock_ref})")
    for child in node.children.values():
        dump(child, depth+1, tok)

def graphviz_dump(node: TreeNode, out_path="kv_tree.dot"):
    dot = Digraph("kv_cache", format="png")
    def visit(n, uid="root"):
        label = "".join(chr(0x2588) for _ in n.key) or "·"
        dot.node(uid, f"{label}\n(len={len(n.key)})")
        for i, c in enumerate(n.children.values()):
            cid = f"{uid}.{i}"
            dot.edge(uid, cid)
            visit(c, cid)
    visit(node)
    base, _ = os.path.splitext(out_path)
    dot.save(out_path)
    dot.render(base, cleanup=True)
    print(f"Graphviz tree rendered to {base}.png")

cache = RadixCache(None, None, page_size=1, disable=False)

with open(OUT_PRED_JSONL, "w", encoding="utf-8") as pred_file, \
     open(KV_STATS_CSV, "w", newline='', encoding="utf-8") as stat_file:
    stat_writer = csv.writer(stat_file)
    stat_writer.writerow(["dialogue_id","turn","node_count","max_depth"])

    with open(PROMPTS_FILE, "r", encoding="utf-8") as pf:
        for dialog in pf:
            data = json.loads(dialog)
            dialogue_id = data["id"]
            turns = [t for t in data["conv"] if t.get("do_inference", False)]

            cache.root_node.children.clear()
            print(f"\n=== Dialogue {dialogue_id} ===")

            for turn_idx, turn in enumerate(turns, start=1):
                user_txt = turn["user"]
                print(f"\n-- Turn {turn_idx}: USER → {user_txt!r}")

                u_ids = tok.encode(user_txt, add_special_tokens=False)
                cache.insert(u_ids)

                max_toks = turn.get("max_tokens", 1024)
                payload = {
                    "model": MODEL_ID,
                    "prompt": user_txt,
                    "max_tokens": max_toks,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "n": 1
                }
                resp = requests.post(
                    f"http://localhost:{SG_LANG_PORT}/v1/completions",
                    json=payload
                )
                reply = resp.json()["choices"][0]["text"].strip()
                print(f"REPLY → {reply!r}")

                r_ids = tok.encode(reply, add_special_tokens=False)
                cache.insert(r_ids)

                print("\nCurrent KV-cache structure (text dump):")
                dump(cache.root_node, tok=tok)

                gv_dot = f"kv_{dialogue_id}_turn{turn_idx}.dot"
                graphviz_dump(cache.root_node, out_path=gv_dot)

                nodes = count_nodes(cache.root_node)
                depth = max_depth(cache.root_node)
                stat_writer.writerow([dialogue_id, turn_idx, nodes, depth])

                pred = {"id": turn["id"], "sys": reply}
                pred_file.write(json.dumps(pred, ensure_ascii=False) + "\n")

turns, sizes = [], []
with open(KV_STATS_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        turns.append(int(row["turn"]))
        sizes.append(int(row["node_count"]))

plt.figure(figsize=(6, 4))
plt.plot(turns, sizes, marker='o')
plt.xlabel("Turn index")
plt.ylabel("KV-tree node count")
plt.title("Refinement Cache Growth (max_tokens=1024)")
plt.tight_layout()
plt.savefig(PLOT_OUT)
print(f"Plot saved to {PLOT_OUT}")
