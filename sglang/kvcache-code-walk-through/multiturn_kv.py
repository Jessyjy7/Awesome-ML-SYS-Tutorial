#!/usr/bin/env python
from sglang.srt.client import SGLangClient
from sglang.srt.mem_cache import tree_cache

def print_tree(node, depth=0):
    indent = "  " * depth
    print(f"{indent}{node.prefix!r} (refs={node.ref_count})")
    for c in getattr(node, "children", []):
        print_tree(c, depth+1)

def chat_and_peek(prompts):
    client = SGLangClient(host="localhost", port=50051)
    for i, p in enumerate(prompts, 1):
        print(f"\n🗨️  Turn {i} → “{p}”")
        resp = client.generate(p)  
        print("🤖", resp.strip())
        print(f"\n🌳 KV-tree after turn {i}:")
        print_tree(tree_cache._root)

if __name__ == "__main__":
    convo = [
        "Hello, how are you?",
        "Tell me a joke.",
        "Now explain why that’s funny.",
        "Tell me another joke",
        "Now explain why that’s funny.",
        "Hello, do you think human beings will move to a different planet?",
        "Hi, how how much is 3pi?"
    ]
    chat_and_peek(convo)
