"""Convert newly collected, table-aligned ShareGPT records
to Qwen3.5 native format with the short SFT prompt and canonical tool schema.
Base-frame/unknown-frame records are rejected, not relabeled or migrated.

Usage:
    python convert_to_native.py can_in_bin
    python convert_to_native.py lemon_in_bin
    python convert_to_native.py <dataset_name>

The script reads:  data/<name>/finetune_data.jsonl
and writes:        data/<name>/finetune_data_native.jsonl
Image paths are preserved; the converter does not move image assets.
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
PROMPTS_DIR = REPO_ROOT / "configs/prompts"


def load_system_prompt(name: str = "sft_v13b_short") -> str:
    """Read prompt text from configs/prompts/<name>.txt — same source
    of truth as guava.inference.prompts so train/serve match byte-for-byte.
    """
    path = name if Path(name).is_absolute() else PROMPTS_DIR / (
        name if name.endswith(".txt") else f"{name}.txt"
    )
    return Path(path).read_text().rstrip()

# The same model-facing signatures used by collection and rollout execution.
# Support direct script execution from any working directory.
sys.path.insert(0, str(REPO_ROOT))
from guava.tools.model_contract import tools_schema, validate_call
TOOLS = tools_schema()


def parse_assistant_turn(value: str) -> dict:
    """Parse a collected turn: '<think>...</think><tool_call>{json}</tool_call>' or
    '<think>...</think>Task complete.'

    Returns a dict with 'reasoning_content', 'content', and optional 'tool_calls',
    suitable for the Qwen3.5 chat_template (which auto-emits <think>, <tool_call> XML).
    """
    m = re.search(r"<think>(.*?)</think>", value, flags=re.DOTALL)
    if not m:
        if value.strip() in ('Task complete.', 'Task failed.'):
            return {'role': 'assistant', 'content': value.strip()}
        raise ValueError(f"assistant turn missing <think>: {value[:200]!r}")
    reasoning = m.group(1).strip()
    rest = (value[: m.start()] + value[m.end() :]).strip()

    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", rest, flags=re.DOTALL)
    if tc_match:
        if rest.count('<tool_call>') != 1 or rest.count('</tool_call>') != 1:
            raise ValueError('Expected exactly one tool call')
        tc = json.loads(tc_match.group(1))
        validate_call(tc["name"], tc.get("arguments", {}))
        content = (rest[: tc_match.start()] + rest[tc_match.end() :]).strip()
        return {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc.get("arguments", {}),
                    },
                }
            ],
        }
    if rest not in ('Task complete.', 'Task failed.'):
        raise ValueError('Expected one tool call or an explicit terminal response')
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": rest,
    }


def parse_human_turn(value: str, images: list[str], img_cursor: list[int]) -> dict:
    has_image = "<image>" in value
    text = value.replace("<image>", "").strip()
    if has_image:
        img = images[img_cursor[0]]
        img_cursor[0] += 1
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }
    return {"role": "user", "content": text}


def parse_tool_turn(value: str, images: list[str], img_cursor: list[int]) -> dict:
    has_image = "<image>" in value
    text = value.replace("<image>", "").strip()
    if has_image:
        img = images[img_cursor[0]]
        img_cursor[0] += 1
        return {
            "role": "tool",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": img},
            ],
        }
    return {"role": "tool", "content": text}


def convert_row(row: dict, fix_image_path, system_prompt: str) -> dict:
    metadata = row.get("metadata", {})
    metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
    if metadata.get("runtime", {}).get("coordinate_frame") != "table_aligned":
        raise ValueError("Only newly collected table_aligned records are supported; missing/base frame is rejected")
    images = [fix_image_path(p) for p in row["images"]]
    img_cursor = [0]

    messages = [{"role": "system", "content": system_prompt}]
    for turn in row["conversations"]:
        role, val = turn["from"], turn["value"]
        if role == "human":
            messages.append(parse_human_turn(val, images, img_cursor))
        elif role == "gpt":
            message = parse_assistant_turn(val)
            if 'loss' in turn:
                message['loss'] = turn['loss']
            messages.append(message)
        elif role == "tool":
            messages.append(parse_tool_turn(val, images, img_cursor))
        else:
            raise ValueError(f"unknown role: {role}")

    if img_cursor[0] != len(images):
        raise ValueError(
            f"image count mismatch for {row['id']}: "
            f"consumed {img_cursor[0]} of {len(images)}"
        )

    return {
        "id": row["id"],
        "messages": messages,
        "tools": TOOLS,
        "images": images,
        "metadata": row["metadata"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_name", help="e.g. can_in_bin or lemon_in_bin")
    parser.add_argument("--input",  type=Path, default=None,
                        help="override input path (default: data/<name>/finetune_data.jsonl)")
    parser.add_argument("--output", type=Path, default=None,
                        help="override output path (default: data/<name>/finetune_data_native.jsonl)")
    parser.add_argument("--system-prompt", default="sft_v13b_short",
                        help="prompt name (resolves configs/prompts/<name>.txt) "
                             "or absolute path. default: sft_v13b_short")
    args = parser.parse_args()

    input_path  = args.input  or DATA_ROOT / args.dataset_name / "finetune_data.jsonl"
    output_path = args.output or DATA_ROOT / args.dataset_name / "finetune_data_native.jsonl"
    fix_image_path = lambda path: path  # New records already reference their image assets.
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"system prompt: {args.system_prompt} ({len(system_prompt)} chars)")

    with input_path.open() as f:
        rows = [json.loads(l) for l in f]
    print(f"loaded {len(rows)} rows from {input_path}")

    converted = []
    failures = []
    for r in rows:
        try:
            converted.append(convert_row(r, fix_image_path, system_prompt))
        except Exception as e:
            failures.append((r["id"], str(e)))

    if failures:
        print(f"\nFAILED {len(failures)} rows:")
        for rid, err in failures[:10]:
            print(f"  {rid}: {err}")
        sys.exit(1)

    with output_path.open("w") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(converted)} rows to {output_path}")


if __name__ == "__main__":
    main()
