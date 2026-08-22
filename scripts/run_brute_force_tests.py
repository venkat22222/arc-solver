import sys
from pathlib import Path

# Add arc-solver and tests to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "tests"))

from test_brute_force_refactored import (
    test_registry_contains_expected_stages,
    test_registry_stage_order,
    test_known_solvable_and_unsolvable_puzzles,
    test_time_budget_respects_limits,
    test_candidate_limits_stop_execution,
    test_simplicity_scoring,
)

if __name__ == "__main__":
    print("Running refactored brute-force tests manually via Python...")
    try:
        test_registry_contains_expected_stages()
        print("1. test_registry_contains_expected_stages: PASS")
        
        test_registry_stage_order()
        print("2. test_registry_stage_order: PASS")
        
        test_known_solvable_and_unsolvable_puzzles()
        print("3. test_known_solvable_and_unsolvable_puzzles: PASS")
        
        test_time_budget_respects_limits()
        print("4. test_time_budget_respects_limits: PASS")
        
        test_candidate_limits_stop_execution()
        print("5. test_candidate_limits_stop_execution: PASS")
        
        test_simplicity_scoring()
        print("6. test_simplicity_scoring: PASS")
        
        print("\nSUCCESS: All brute-force refactor tests passed successfully!")
    except AssertionError as e:
        print(f"\nFAILURE: Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: Error running tests: {e}")
        sys.exit(1)
