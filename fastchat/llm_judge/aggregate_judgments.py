"""Aggregate pairwise judgments from independent judge models by majority vote.

Example:
python aggregate_judgments.py --input-files gpt-4o_pair.jsonl claude_pair.jsonl llama_pair.jsonl \
    --output-file ensemble_pair.jsonl
"""
import argparse
from collections import Counter
import json
import os


def _strict_winner(record):
    """Return a model-relative vote, treating position disagreement as a tie."""
    first = record.get("g1_winner", "error")
    second = record.get("g2_winner", "error")
    if first == second and first in ("model_1", "model_2", "tie"):
        return first
    return "tie"


def aggregate_judgments(input_files, output_file):
    if len(input_files) < 3:
        raise ValueError("Provide at least three judge files for majority voting.")

    judges = []
    for filename in input_files:
        records = {}
        with open(filename, "r", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                key = (
                    record["question_id"],
                    record["model_1"],
                    record["model_2"],
                    record.get("turn", 1),
                )
                records[key] = record
        judges.append((filename, records))

    common_keys = set.intersection(*(set(records) for _, records in judges))
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as output:
        for key in sorted(common_keys, key=lambda item: (item[0], item[3], item[1], item[2])):
            votes = {
                os.path.basename(filename): _strict_winner(records[key])
                for filename, records in judges
            }
            counts = Counter(votes.values())
            winner, count = counts.most_common(1)[0]
            if count < len(judges) // 2 + 1:
                winner = "tie"
            question_id, model_1, model_2, turn = key
            output.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "model_1": model_1,
                        "model_2": model_2,
                        "turn": turn,
                        "judge": ["ensemble", "majority-vote"],
                        "winner": winner,
                        "g1_winner": winner,
                        "g2_winner": winner,
                        "judge_votes": votes,
                        "g1_judgment": "Aggregated from independent judge votes.",
                        "g2_judgment": "Aggregated from independent judge votes.",
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-files", nargs="+", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()
    aggregate_judgments(args.input_files, args.output_file)
