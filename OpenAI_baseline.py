import os
import json
import asyncio
import time
import random
import unicodedata
from tqdm.asyncio import tqdm_asyncio
from openai import AsyncOpenAI
import openai

# 🔐 API key
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📂 Paths
BASE_PATH = "/Users/brunosotic/Desktop/Eloquent 2026/data"
RESULTS_PATH = "/Users/brunosotic/Desktop/Eloquent 2026/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

FILES = [
    "de_specific.jsonl",
    "de_unspecific.jsonl",
    "en_specific.jsonl",
    "en_unspecific.jsonl",
    "fr_specific.jsonl",
    "fr_unspecific.jsonl",
    "it_specific.jsonl",
    "it_unspecific.jsonl",
    "sv_specific.jsonl",
    "sv_unspecific.jsonl",
]

# 🔁 Settings
N_RUNS = 15
MODEL = "gpt-4.1-mini"
MODEL_TAG = "gpt41mini"
TEMPERATURE = 0
MAX_TOKENS = 200
CONCURRENCY = 20


# ----------------------------
# Utils
# ----------------------------

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_completed(path):
    """Read completed (id, run) pairs — tolerates partial/corrupt lines."""
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "run" in row and "id" in row and "answer" in row:
                    completed.add((str(row["id"]), int(row["run"])))
            except Exception:
                continue  # skip malformed lines silently
    return completed


write_lock = asyncio.Lock()

async def append_jsonl(path, row):
    async with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sanitize(prompt):
    prompt = prompt.replace("\x00", "")
    prompt = "".join(c for c in prompt if unicodedata.category(c) != "Cc")
    prompt = unicodedata.normalize("NFC", prompt)
    return prompt


async def query_model(prompt, retries=20):
    prompt = sanitize(prompt)
    wait = 1
    for attempt in range(retries):
        try:
            await asyncio.sleep(random.uniform(0, 0.1))
            response = await client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            print(f"\n⏳ Rate limit hit, retrying in {wait}s...")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 60)
        except openai.BadRequestError as e:
            # 400 errors are transient under async load — just retry
            await asyncio.sleep(random.uniform(0.5, 1.5))
        except openai.APIError as e:
            print(f"\n⚠️  API error: {e}, retrying in {wait}s...")
            await asyncio.sleep(wait)
            wait = min(wait * 2, 60)
    raise Exception(f"Failed after {retries} attempts")


def parse_filename(filename):
    lang, typ = filename.replace(".jsonl", "").split("_")
    return lang, typ


# ----------------------------
# Run per file
# ----------------------------

async def run_file(filename):
    input_path = os.path.join(BASE_PATH, filename)
    data = load_jsonl(input_path)
    lang, typ = parse_filename(filename)
    output_filename = f"{MODEL_TAG}_{lang}_{typ}.jsonl"
    output_path = os.path.join(RESULTS_PATH, output_filename)

    completed = load_completed(output_path)
    if completed:
        print(f"⏭️  Resuming: {filename} ({len(completed)} calls already done)")

    all_tasks = [
        (item, run_idx)
        for item in data
        for run_idx in range(N_RUNS)
        if (str(item["id"]), run_idx) not in completed
    ]

    total_calls = len(data) * N_RUNS
    remaining = len(all_tasks)

    print(f"\n🚀 Running: {filename}")
    print(f"➡️  Output: {output_filename}")
    print(f"Prompts: {len(data)} | Runs: {N_RUNS} | Total: {total_calls} | Remaining: {remaining}")

    if not all_tasks:
        print("✅ Already complete, skipping.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    failed = []

    async def run_single(item, run_idx, pbar):
        async with semaphore:
            try:
                answer = await query_model(item["prompt"])
                await append_jsonl(output_path, {
                    "id": item["id"],
                    "run": run_idx,
                    "prompt": item["prompt"],
                    "answer": answer,
                })
            except Exception as e:
                print(f"\n❌ Queuing for retry — id={item['id']} run={run_idx}: {e}")
                failed.append((item, run_idx))
            finally:
                pbar.update(1)

    # --- First pass ---
    with tqdm_asyncio(total=remaining, desc=filename) as pbar:
        tasks = [run_single(item, run_idx, pbar) for item, run_idx in all_tasks]
        await asyncio.gather(*tasks)

    # --- Retry loop ---
    pass_num = 1
    while failed:
        pass_num += 1
        retry_batch = failed.copy()
        failed.clear()
        print(f"\n🔁 Retry pass #{pass_num}: {len(retry_batch)} calls remaining...")
        await asyncio.sleep(10)
        with tqdm_asyncio(total=len(retry_batch), desc=f"{filename} retry#{pass_num}") as pbar:
            tasks = [run_single(item, run_idx, pbar) for item, run_idx in retry_batch]
            await asyncio.gather(*tasks)

    print(f"✅ Done: {output_path}")


# ----------------------------
# Main
# ----------------------------

async def main():
    print(f"\n🤖 Model: {MODEL}  |  Runs per prompt: {N_RUNS}  |  Concurrency: {CONCURRENCY}")
    print(f"📁 Files: {len(FILES)}\n")
    start = time.time()

    for file in FILES:
        await run_file(file)

    elapsed = time.time() - start
    print(f"\n🏁 All done in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())