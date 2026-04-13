import os
import json
import time
from tqdm import tqdm
from google import genai

# 🔐 API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

BASE_PATH = "/Users/brunosotic/Desktop/Eloquent 2026/data"
RESULTS_PATH = "/Users/brunosotic/Desktop/Eloquent 2026/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

INPUT_FILE = "en_unspecific.jsonl"
OUTPUT_FILE = "TEST_gemini_en_unspecific.jsonl"

N_PROMPTS = 10
N_RUNS = 30

REQUEST_DELAY = 0.5

# Latest stable model (use "gemini-3-flash-preview" for bleeding edge)
MODEL = "gemini-2.5-flash"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def query_model(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    return response.text.strip()


def main():
    input_path = os.path.join(BASE_PATH, INPUT_FILE)
    output_path = os.path.join(RESULTS_PATH, OUTPUT_FILE)

    data = load_jsonl(input_path)[:N_PROMPTS]

    total_calls = len(data) * N_RUNS  # safer than N_PROMPTS * N_RUNS if file is shorter

    print("\n🧪 GEMINI TEST RUN")
    print(f"Model:   {MODEL}")
    print(f"Prompts: {len(data)} | Runs: {N_RUNS}")
    print(f"Total calls: {total_calls}\n")

    with tqdm(total=total_calls) as pbar:
        for item in data:
            prompt_id = item["id"]
            prompt = item["prompt"]

            for run_idx in range(N_RUNS):
                try:
                    answer = query_model(prompt)

                    append_jsonl(output_path, {
                        "id": prompt_id,
                        "run": run_idx,
                        "prompt": prompt,
                        "answer": answer,
                    })

                    time.sleep(REQUEST_DELAY)

                except Exception as e:
                    print(f"\n⚠️  Error on id={prompt_id} run={run_idx}: {e}")
                    time.sleep(5)

                pbar.update(1)

    print(f"\n✅ Done → {output_path}")


if __name__ == "__main__":
    main()