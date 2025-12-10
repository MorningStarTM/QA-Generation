import os
import json
from pathlib import Path
from src.utils.logger import logger

def save_qa_from_completions(completions, output_dir="output", filename="qa.json"):
    """
    completions: an object like
        Completions(qa_json=['[{"question": "...", "answer": "..."}, ...]'])
    """
    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    all_qa = []

    # completions.qa_json is a list of JSON strings
    for qa_block in completions.qa_json:
        try:
            data = json.loads(qa_block)
        except json.JSONDecodeError as e:
            logger.info(f"Failed to parse JSON block: {e}")
            continue

        # Normalize to list
        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            for item in data:
                q = item.get("question")
                a = item.get("answer")
                if q is not None and a is not None:
                    all_qa.append({"question": q, "answer": a})

    # Save to output/qa.json
    out_path = Path(output_dir) / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(all_qa)} Q&A pairs to {out_path}")
    return str(out_path)