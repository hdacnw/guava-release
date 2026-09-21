#!/usr/bin/env python
"""Native 格式 -> ms-swift 期望的格式：
- content 字段统一为 string
- inline {"type":"image"} 块替换为 <image> 占位符
- 顶层 images 列表保留（路径与 <image> 出现顺序一一对应）
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

n_in = n_out = n_img_total = 0
with src.open() as f, dst.open("w") as out:
    for line in f:
        if not line.strip():
            continue
        n_in += 1
        rec = json.loads(line)
        n_imgs = 0
        for msg in rec.get("messages", []):
            c = msg.get("content")
            if isinstance(c, list):
                parts = []
                for block in c:
                    btype = block.get("type")
                    if btype == "image":
                        parts.append("<image>")
                        n_imgs += 1
                    elif btype == "text":
                        parts.append(block.get("text", ""))
                msg["content"] = "".join(parts)
            # already string -> leave as-is
        n_img_total += n_imgs
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_out += 1

print(f"normalized {n_out}/{n_in} records, total {n_img_total} <image> placeholders  ->  {dst}")
