#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a bilingual dialogue script (bilingual_docs.md) from:
  - english.md  (multi-speaker transcript)
  - dictionary.md (markdown table with Number | English | Chinese-without-punctuation)

Usage:
  python build_bilingual.py \
    --english english.md \
    --dictionary dictionary.md \
    --output bilingual_docs.md \
    --model gpt-4o-mini \
    --min-sim 0.35

Environment:
  OPENAI_API_KEY must be set.

Notes:
  - Idempotent if run with temperature=0.
  - Emits alignment ranges to alignment.json for auditing.
"""

import os
import re
import json
import argparse
from difflib import SequenceMatcher

# ---------- I/O helpers ----------

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# ---------- dictionary.md parsing ----------

def parse_dictionary(dict_lines):
    """
    Parse a Markdown table with header:
      | Number | English | Chinese |
    Returns sorted list of rows: {num, en, zh_no_punct}
    """
    rows = []
    for line in dict_lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        # skip header
        low = s.lower().replace(" ", "")
        if low.startswith("|number|english|chinese|") or low.startswith("|---|---|---|"):
            continue
        # split cells
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3:
            continue
        num_cell = cells[0]
        if not num_cell.isdigit():
            # sometimes # or other chars — try to coerce
            try:
                _ = int(re.sub(r"[^\d]", "", num_cell))
                num_cell = _
            except Exception:
                continue
        num = int(num_cell)
        en = cells[1]
        zh = cells[2]
        rows.append({"num": num, "en": en, "zh_no_punct": zh})
    rows.sort(key=lambda r: r["num"])
    return rows

# ---------- english.md parsing ----------

SPEAKER_RE = re.compile(r"^([^\n:]+):\s*(.*)$")

def parse_english_paragraphs(lines):
    """
    A paragraph starts when a line matches 'Speaker: ...'
    Consecutive lines until the next speaker label belong to that speaker paragraph.
    Lines without a speaker label are passed through as raw lines.
    """
    paragraphs = []
    curr = None
    for raw in lines:
        line = raw.rstrip("\n")
        m = SPEAKER_RE.match(line)
        if m:
            # flush previous paragraph
            if curr:
                paragraphs.append(curr)
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            curr = {"speaker": speaker, "lines": [text] if text else []}
        else:
            if curr and line.strip() != "":
                curr["lines"].append(line.strip())
            else:
                # standalone raw (headings, blanks, narration)
                paragraphs.append({"speaker": None, "lines": [line]})
    if curr:
        paragraphs.append(curr)
    return paragraphs

# ---------- alignment ----------

def similar(a, b):
    return SequenceMatcher(None, a or "", b or "").ratio()

def align_paragraphs(paragraphs, dict_rows, min_first_last_sim=0.35):
    """
    Maintains a forward-only pointer into dict_rows.
    For each speaker paragraph, grabs consecutive dict rows until cumulative EN length
    roughly matches paragraph length. Non-speaker lines are passed through.
    """
    aligned = []
    dict_ptr = 0
    N = len(dict_rows)

    for p in paragraphs:
        if p["speaker"] is None:
            aligned.append({"type": "raw", "text": "\n".join(p["lines"])})
            continue

        en_text = " ".join([s for s in p["lines"] if s]).strip()
        if not en_text:
            aligned.append({
                "type": "speaker",
                "speaker": p["speaker"],
                "en": "",
                "dict_start": None,
                "dict_end": None
            })
            continue

        # Heuristic: consume dict rows until length ~ matches
        start = dict_ptr
        i = start
        accum_en = []
        target_len = max(len(en_text), 1)
        while i < N and len(" ".join(accum_en)) < target_len * 0.95:
            accum_en.append(dict_rows[i]["en"])
            i += 1
        end = max(i - 1, start)

        # Optional sanity (non-blocking)
        first_ok = True
        last_ok = True
        if start < N and p["lines"]:
            first_ok = similar(dict_rows[start]["en"], p["lines"][0]) >= min_first_last_sim
            last_ok  = similar(dict_rows[end]["en"],   p["lines"][-1]) >= min_first_last_sim

        aligned.append({
            "type": "speaker",
            "speaker": p["speaker"],
            "en": en_text,
            "dict_start": start,
            "dict_end": end,
            "first_ok": first_ok,
            "last_ok": last_ok
        })
        dict_ptr = end + 1

    return aligned

# ---------- Chinese collection & punctuation ----------

def collect_zh(dict_rows, start, end):
    if start is None or end is None or start < 0 or end < start:
        return ""
    parts = []
    for r in dict_rows[start:end+1]:
        if r["zh_no_punct"]:
            parts.append(r["zh_no_punct"])
    return " ".join(parts).strip()

CHN_FULLWIDTH_COLON = "："

def render_block(speaker_en, text_en, speaker_map, text_zh_punct):
    speaker_zh = speaker_map.get(speaker_en, speaker_en)
    # Bold names; colon for EN, full-width colon for ZH
    return (
        f"**{speaker_en}:** {text_en}\n"
        f"**{speaker_zh}{CHN_FULLWIDTH_COLON}**{text_zh_punct}\n"
    )

# ---------- OpenAI punctuation call ----------

def punctuate_batch_openai(texts, model):
    """
    Batch punctuation for Chinese strings separated by a unique delimiter.
    Returns a list of punctuated strings in the same order.
    """
    if not texts:
        return []

    sep = "<<<SEP>>>"
    payload = "\n" + sep + "\n"
    prompt = (f"你是严格的中文文字编辑。只为给定文本添加标准中文标点符号；"
              f"不要改动任何字词，不要增删词语，不要改变顺序。"
              f"输入中将包含多段文本，以分隔符 {sep} 分隔。逐段输出，对应位置输出，"
              f"不要输出任何额外解释，仍用 {sep} 分隔。")

    joined = f"{sep}".join([t if t is not None else "" for t in texts])

    # Lazy import to avoid hard dep if user just wants dry-run
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system",
             "content": "You are a strict Chinese copy editor. Only add standard Chinese punctuation to the provided text. "
                        "Do NOT change wording; do NOT add or remove words; do NOT reorder text. "
                        f"Multiple segments are separated by {sep}. Return the outputs joined by {sep} in the same order."},
            {"role": "user", "content": f"{prompt}\n\n{joined}"}
        ]
    )
    content = resp.choices[0].message.content.strip()
    # Robust split (model may add newlines around)
    parts = [p.strip() for p in content.split(sep)]
    # If mismatch, pad/truncate gracefully
    if len(parts) != len(texts):
        # Try a fallback naive approach: return original texts as-is for safety
        # (You can swap to raise an error if you prefer strictness.)
        return [p if i < len(parts) else (texts[i] or "")
                for i, p in enumerate(parts + [""] * (len(texts) - len(parts)))]
    return parts

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Build bilingual_docs.md from english.md + dictionary.md")
    parser.add_argument("--english", "-e", default="english.md")
    parser.add_argument("--dictionary", "-d", default="dictionary.md")
    parser.add_argument("--output", "-o", default="bilingual_docs.md")
    parser.add_argument("--model", "-m", default="gpt-4o-mini")
    parser.add_argument("--min-sim", type=float, default=0.35)
    parser.add_argument("--speaker-map", default=None, help="Path to JSON mapping English->Chinese names")
    parser.add_argument("--emit-alignment", default="alignment.json")
    args = parser.parse_args()

    # Speaker map (extend as needed; or pass JSON path)
    SPEAKER_MAP = {
        "Johnathan Bi": "毕英杰",
        "David O’Connor": "大卫·奥康纳",
        "David O'Connor": "大卫·奥康纳",  # normalize curly/straight apostrophes
    }
    if args.speaker_map and os.path.exists(args.speaker_map):
        try:
            SPEAKER_MAP.update(json.loads(read_text_file(args.speaker_map)))
        except Exception:
            pass

    # Read inputs
    english_lines = read_text_file(args.english).splitlines()
    dict_lines = read_text_file(args.dictionary).splitlines()

    # Parse dictionary
    DICT = parse_dictionary(dict_lines)
    if not DICT:
        raise RuntimeError("No dictionary rows parsed. Check dictionary.md formatting.")

    # Parse english paragraphs
    PARAS = parse_english_paragraphs(english_lines)

    # Align
    ALIGNED = align_paragraphs(PARAS, DICT, min_first_last_sim=args.min_sim)

    # Prepare zh batch
    zh_batches = []
    aligned_indices = []
    for i, node in enumerate(ALIGNED):
        if node["type"] == "speaker":
            zh_no_punct = collect_zh(DICT, node["dict_start"], node["dict_end"])
            zh_batches.append(zh_no_punct)
            aligned_indices.append(i)

    # Punctuate (API)
    puncted = punctuate_batch_openai(zh_batches, model=args.model) if zh_batches else []

    # Assemble output
    out_lines = []
    z = 0
    for node in ALIGNED:
        if node["type"] == "raw":
            out_lines.append(node["text"])
            continue
        # speaker block
        text_en = node["en"]
        text_zh_punct = puncted[z] if z < len(puncted) else ""
        z += 1
        block = render_block(node["speaker"], text_en, SPEAKER_MAP, text_zh_punct)
        out_lines.append(block)
        out_lines.append("")  # blank line between paragraphs

    out = "\n".join(out_lines).strip() + "\n"
    write_text_file(args.output, out)

    # Emit alignment audit
    audit = []
    for node in ALIGNED:
        if node["type"] == "speaker":
            audit.append({
                "speaker": node["speaker"],
                "dict_start": node["dict_start"],
                "dict_end": node["dict_end"],
                "first_ok": node.get("first_ok"),
                "last_ok": node.get("last_ok"),
                "en_preview": (node["en"][:160] + "…") if len(node["en"]) > 160 else node["en"]
            })
        else:
            audit.append({"raw": node["text"][:120]})
    write_text_file(args.emit_alignment, json.dumps(audit, ensure_ascii=False, indent=2))

    print(f"✅ Wrote: {args.output}")
    print(f"🧭 Alignment audit: {args.emit_alignment}")

if __name__ == "__main__":
    main()
