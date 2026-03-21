# 18.09.2026
# Data scaper

import os
import re
from datasets import load_dataset

SAVE_PATH  = "data/training.txt"
SEPARATOR  = "\n\n"
MIN_REPLY_LEN = 20

# clean from trash chars, or will slow down training

ALLOWED       = re.compile(r"[^\x20-\x7E\n\t]")
MULTI_NEWLINE = re.compile(r"\n{3,}")
MULTI_SPACE   = re.compile(r"[ \t]{2,}")

REPLACEMENTS = str.maketrans({
    "\u2018": "'",  "\u2019": "'",  "\u201c": '"',  "\u201d": '"',
    "\u2013": "-",  "\u2014": "-",  "\u2026": "...",
    "\u00e9": "e",  "\u00e8": "e",  "\u00ea": "e",  "\u00eb": "e",
    "\u00e0": "a",  "\u00e2": "a",  "\u00e4": "a",  "\u00f6": "o",
    "\u00fc": "u",  "\u00f1": "n",  "\u00df": "ss", "\u00e7": "c",
    "\u00e1": "a",  "\u00ed": "i",  "\u00f3": "o",  "\u00fa": "u",
    "\u00b4": "'",  "\u0060": "'",  "\u2022": "-",  "\u00a0": " ",
    "\u00b7": "-",  "\u2012": "-",  "\u2015": "-",  "\u00ad": "",
    "\u200b": "",   "\u200c": "",   "\u200d": "",   "\ufeff": "",
})


def clean(text: str) -> str:
    text = text.translate(REPLACEMENTS)         # fancy chars to ASCII
    text = ALLOWED.sub("", text)                # drop anything still non ASCII
    text = MULTI_NEWLINE.sub("\n\n", text)      # normalise
    text = MULTI_SPACE.sub(" ", text)           # normalise
    return text.strip()

def build_threads(messages: list) -> list[list[dict]]:

    by_id     = {m["message_id"]: m for m in messages}
    children  = {}
    roots     = []

    for m in messages:
        pid = m.get("parent_id")
        if pid is None:
            roots.append(m["message_id"])
        else:
            children.setdefault(pid, []).append(m["message_id"])

    def walk(mid, thread):
        m    = by_id[mid]
        role = "User" if m["role"] == "prompter" else "Assistant"
        text = clean(m.get("text", ""))
        if len(text) >= MIN_REPLY_LEN:
            thread.append({"role": role, "text": text})
            
        # follow highest ranked child
        
        kids = children.get(mid, [])
        if kids:
            best = max(kids, key=lambda k: by_id[k].get("rank", 0) or 0)
            walk(best, thread)

    threads = []
    for root in roots:
        thread = []
        walk(root, thread)
        
        # only keep threads that have at least one user
        
        if len(thread) >= 2:
            threads.append(thread)

    return threads


def thread_to_text(thread: list[dict]) -> str:
    """Format a thread as plain dialogue text."""
    lines = []
    for turn in thread:
        lines.append(f"{turn['role']}: {turn['text']}")
    return "\n".join(lines)


# download

ds = load_dataset("OpenAssistant/oasst1", split="train")

# Filter to english only

english = [m for m in ds if m.get("lang", "en") == "en"]
print(f"Total messages: {len(ds):,}  |  English only: {len(english):,}\n")

threads = build_threads(english)
print(f"Built {len(threads))

os.makedirs("data", exist_ok=True)

kept    = 0
skipped = 0

print(f"Saving to {SAVE_PATH}...")
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    for thread in threads:
        text = thread_to_text(thread)
        if len(text) < 50:
            skipped += 1
            continue
        f.write(text)
        f.write(SEPARATOR)
        kept += 1

size_mb      = os.path.getsize(SAVE_PATH) / 1_000_000
unique_chars = sorted(set(open(SAVE_PATH, encoding="utf-8").read()))

print(f"  Conversations kept:  {kept:,}")
print("─" * 40)
if threads:
    print(thread_to_text(threads[0])[:300])
print("─" * 40)