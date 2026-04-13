import os
import json
import time
from tqdm import tqdm
from openai import OpenAI

# 🔐 API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BASE_PATH = "/Users/brunosotic/Desktop/Eloquent 2026/data"

INPUT_FILE = "en_unspecific.jsonl"   # change to specific if you want
OUTPUT_FILE = "TEST_gpt41mini_en_unspecific.jsonl"

N_PROMPTS = 10
N_RUNS = 30

MODEL = "gpt-4.1-mini"
TEMPERATURE = 0
MAX_TOKENS = 200
REQUEST_DELAY = 0.5


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def query_model(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def main():
    input_path = os.path.join(BASE_PATH, INPUT_FILE)
    output_path = os.path.join(BASE_PATH, OUTPUT_FILE)

    data = load_jsonl(input_path)[:N_PROMPTS]

    total_calls = N_PROMPTS * N_RUNS

    print(f"\n🧪 TEST RUN")
    print(f"Prompts: {N_PROMPTS} | Runs each: {N_RUNS}")
    print(f"Total API calls: {total_calls}\n")

    with tqdm(total=total_calls) as pbar:
        for item in data:
            prompt_id = item["id"]
            prompt = item["prompt"]

            for _ in range(N_RUNS):
                try:
                    answer = query_model(prompt)

                    append_jsonl(output_path, {
                        "id": prompt_id,
                        "prompt": prompt,
                        "answer": answer
                    })

                    time.sleep(REQUEST_DELAY)

                except Exception as e:
                    print(f"⚠️ Error: {e}")
                    time.sleep(5)

                pbar.update(1)

    print(f"\n✅ Test complete → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()