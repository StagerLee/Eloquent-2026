import json
import ollama

def run_file(input_path, output_path):

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:

        for line in infile:
            item = json.loads(line)

            prompt = item["prompt"]

            response = ollama.generate(
                model="mistral",
                prompt=prompt,
                options={
                    "temperature": 0,
                    "num_predict": 200
                }
            )

            answer = response["response"].strip()

            output = {
                "id": item["id"],
                "prompt": prompt,
                "answer": answer
            }

            outfile.write(json.dumps(output) + "\n")


run_file("data/en_unspecific.jsonl", "results/en_unspecific.jsonl")
run_file("data/en_specific.jsonl", "results/en_specific.jsonl")