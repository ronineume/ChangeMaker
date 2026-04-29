#!/usr/bin/env python3
"""
Extract consistency scores from selfcheck_results JSON files to CSV.
Consistency = 1 - AvgHallucinationScore (lower hallucination = higher consistency)
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict


def extract_consistency_to_csv(input_dir: str, output_file: str):
    """
    Extract problem_id, rewrite_type, and consistency scores from JSON files.
    """
    input_path = Path(input_dir)
    results = []
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            problem_id = data.get("problem_id", "")
            rewrite_type = data.get("rewrite_type", "")
            avg_hallucination = data.get("avg_hallucination_score", None)
            
            if avg_hallucination is not None:
                consistency_score = 1.0 - avg_hallucination
                results.append({
                    "ProblemID": problem_id,
                    "RewriteType": rewrite_type,
                    "AvgHallucinationScore": round(avg_hallucination, 4),
                    "ConsistencyScore": round(consistency_score, 4)
                })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Sort by ProblemID then RewriteType
    results.sort(key=lambda x: (x["ProblemID"], x["RewriteType"]))
    
    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["ProblemID", "RewriteType", "AvgHallucinationScore", "ConsistencyScore"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total rows: {len(results)}")
    
    # Print statistics
    print("\n=== Statistics ===")
    rewrite_types = defaultdict(list)
    for r in results:
        rewrite_types[r["RewriteType"]].append(r["ConsistencyScore"])
    
    print(f"{'RewriteType':<35} {'Count':<8} {'AvgConsistency':<15}")
    print("-" * 60)
    for rt in sorted(rewrite_types.keys()):
        scores = rewrite_types[rt]
        avg_consistency = sum(scores) / len(scores)
        print(f"{rt:<35} {len(scores):<8} {avg_consistency:<15.4f}")


if __name__ == "__main__":
    INPUT_DIRECTORY = "e:/Work_Station/Hallu/QA/selfcheck_results"
    OUTPUT_CSV = "e:/Work_Station/Hallu/QA/consistency_scores.csv"
    
    extract_consistency_to_csv(INPUT_DIRECTORY, OUTPUT_CSV)
