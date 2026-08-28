import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Add arc-solver to Python path if running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_puzzles_from_dir
from src.brute_force import try_brute_force


def run_eval():
    parser = argparse.ArgumentParser(description="Evaluate the brute-force solver.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="arc-solver/data/arc-agi-2",  # Default if data is nested
        help="Directory containing the puzzle JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/brute_force_eval.json",
        help="Path to write the JSON report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading puzzles from {data_dir}...")
    puzzles = load_puzzles_from_dir(data_dir)
    print(f"Loaded {len(puzzles)} puzzles.")

    results = []
    total_time_ms = 0.0
    max_time_ms = 0.0
    solved_count = 0
    false_positives = 0
    solves_by_stage = Counter()
    candidate_names_counter = Counter()
    total_simplicity_score = 0.0
    
    false_positives_by_stage = Counter()
    false_positives_by_name = Counter()

    for puzzle in puzzles:
        start_time = time.time()
        hit = try_brute_force(puzzle.train_pairs)
        elapsed_ms = (time.time() - start_time) * 1000.0
        total_time_ms += elapsed_ms
        max_time_ms = max(max_time_ms, elapsed_ms)

        solved = hit is not None
        candidate_name = hit.name if solved else None
        stage = getattr(hit, 'stage', 0) if solved else None
        simplicity_score = getattr(hit, 'simplicity_score', 0.0) if solved else None
        params = getattr(hit, 'params', {}) if solved else None
        warnings = getattr(hit, 'warnings', []) if solved else []

        has_test_outputs = len(puzzle.test_outputs) > 0
        correct_on_test = None
        false_positive = False

        if solved:
            solved_count += 1
            stage = getattr(hit, 'stage', 0)
            solves_by_stage[stage] += 1
            candidate_names_counter[candidate_name] += 1
            total_simplicity_score += simplicity_score

            # Run on test inputs to verify test correctness
            # We construct a namespace with our helper library primitives and execute the brute force code
            # But wait! A simpler, safer way is to just execute the generated code or use our library primitives directly
            # Since the code is formatted as a standard python function "def solve(grid):",
            # we can safely exec() it and call solve(test_input)
            try:
                namespace = {}
                # Inject library helpers into namespace
                from src.library import get_sandbox_helpers
                namespace.update(get_sandbox_helpers())
                exec(hit.code, namespace)
                test_input_copy = [row[:] for row in puzzle.test_inputs[0]]
                actual_test_out = namespace["solve"](test_input_copy)
                
                if has_test_outputs:
                    expected_test_out = puzzle.test_outputs[0]
                    correct_on_test = (actual_test_out == expected_test_out)
                    if not correct_on_test:
                        false_positive = True
                        false_positives += 1
                        false_positives_by_stage[stage] += 1
                        false_positives_by_name[candidate_name] += 1
                    else:
                        false_positive = False
            except Exception as e:
                print(f"Error running solved brute-force code for puzzle {puzzle.id}: {e}")
                correct_on_test = False
                false_positive = True
                false_positives += 1
                false_positives_by_stage[stage] += 1
                false_positives_by_name[candidate_name] += 1

        record = {
            "puzzle_id": puzzle.id,
            "solved_by_brute_force": solved,
            "brute_force_candidate_name": candidate_name,
            "brute_force_stage": stage,
            "time_ms": elapsed_ms,
            "train_pass_count": len(puzzle.train_pairs) if solved else 0,
            "has_test_outputs": has_test_outputs,
            "correct_on_test": correct_on_test,
            "false_positive": false_positive,
            "simplicity_score": simplicity_score,
            "params": params,
            "warnings": warnings,
        }
        results.append(record)

    # Write JSON report
    report = {
        "summary": {
            "total_puzzles": len(puzzles),
            "brute_force_solved_count": solved_count,
            "false_positives_count": false_positives,
            "average_brute_force_time_ms": total_time_ms / len(puzzles) if puzzles else 0.0,
            "average_simplicity_score": total_simplicity_score / solved_count if solved_count else 0.0,
            "solves_by_stage": dict(solves_by_stage),
            "false_positives_by_stage": dict(false_positives_by_stage),
            "false_positives_by_candidate_name": dict(false_positives_by_name),
            "top_candidates": dict(candidate_names_counter.most_common(5)),
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 40)
    print("BRUTE-FORCE EVALUATION SUMMARY")
    print("=" * 40)
    print(f"Total Puzzles Checked:         {len(puzzles)}")
    print(f"Solved by Brute-force:         {solved_count}")
    print(f"False Positives:               {false_positives}")
    print(f"Average Brute-force Time (ms): {total_time_ms / len(puzzles) if puzzles else 0.0:.2f}")
    print(f"Max Brute-force Time (ms):     {max_time_ms:.2f}")
    print(f"Average Simplicity Score:      {total_simplicity_score / solved_count if solved_count else 0.0:.2f}")
    print("\nSolves by Stage:")
    for stage_id in sorted(solves_by_stage.keys()):
        print(f"  Stage {stage_id}: {solves_by_stage[stage_id]}")
    print("\nFalse Positives by Stage:")
    for stage_id in sorted(false_positives_by_stage.keys()):
        print(f"  Stage {stage_id}: {false_positives_by_stage[stage_id]}")
    print("\nFalse Positives by Candidate Name:")
    for name, fp_count in false_positives_by_name.items():
        print(f"  {name}: {fp_count}")
    print("\nTop Successfully Used Candidates:")
    for name, count in candidate_names_counter.most_common(5):
        print(f"  {name}: {count}")
    print("=" * 40)
    print(f"Full report written to {output_path}")


if __name__ == "__main__":
    run_eval()
