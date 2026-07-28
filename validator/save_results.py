#!/usr/bin/env python3
"""
Save validation results for tracking algorithm improvements.
"""

import json
import os
from datetime import datetime


def make_json_serializable(obj):
    """Convert ValidationMetrics objects to dicts for JSON serialization."""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)

def save_validation_results(results: dict, filename: str = None) -> str:
    """
    Save validation results to a JSON file.
    
    Args:
        results: The validation results dict
        filename: Optional custom filename, defaults to timestamp
    
    Returns:
        Path to saved file
    """
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate filename with timestamp
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{timestamp}.json"
    
    filepath = os.path.join(reports_dir, filename)
    
    # Add metadata from current algorithm state
    state_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'algorithm_state.json')
    algo_version = 'unknown'
    weights = {}
    try:
        with open(state_path) as f:
            state = json.load(f)
            algo_version = f"v{state.get('version', 'unknown')}"
            weights = state.get('weights', {})
    except (OSError, json.JSONDecodeError):
        pass

    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'algorithm_version': algo_version,
        'weights': weights,
    }
    
    # Save to file - convert ValidationMetrics to dicts
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=make_json_serializable)
    
    print(f"Results saved to: {filepath}")
    return filepath


def load_validation_results(filename: str) -> dict:
    """Load validation results from a JSON file."""
    filepath = os.path.join("reports", filename)
    with open(filepath, 'r') as f:
        return json.load(f)


def list_saved_results() -> list:
    """List all saved validation results."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return []
    
    files = [f for f in os.listdir(reports_dir) if f.startswith('validation_') and f.endswith('.json')]
    files.sort(reverse=True)  # Newest first
    return files


def compare_results(filename1: str, filename2: str):
    """
    Compare two validation results to see improvements.
    """
    results1 = load_validation_results(filename1)
    results2 = load_validation_results(filename2)
    
    print(f"\nComparing {filename1} vs {filename2}")
    print("=" * 60)
    
    metrics = [
        ('Overall Exact Match Rate', 'overall_exact_match_rate'),
        ('Overall Top 3 Accuracy', 'overall_top3_accuracy'),
        ('Overall Mean Error', 'overall_mae')
    ]
    
    for label, key in metrics:
        val1 = results1.get(key, 0)
        val2 = results2.get(key, 0)
        
        if key == 'overall_mae':
            # Lower is better for MAE
            change = val1 - val2
            direction = '↓' if change > 0 else '↑'
        else:
            # Higher is better for percentages
            change = val2 - val1
            direction = '↑' if change > 0 else '↓'
        
        print(f"{label}:")
        print(f"  {filename1}: {val1:.1%}" if 'rate' in key or 'accuracy' in key else f"  {filename1}: {val1:.2f}")
        print(f"  {filename2}: {val2:.1%}" if 'rate' in key or 'accuracy' in key else f"  {filename2}: {val2:.2f}")
        print(f"  Change: {direction} {abs(change):.1%}" if 'rate' in key or 'accuracy' in key else f"  Change: {direction} {abs(change):.2f}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage validation results')
    parser.add_argument('--list', action='store_true', help='List saved results')
    parser.add_argument('--compare', nargs=2, metavar=('FILE1', 'FILE2'), help='Compare two results')
    args = parser.parse_args()
    
    if args.list:
        files = list_saved_results()
        print("Saved validation results:")
        for f in files:
            print(f"  - {f}")
    elif args.compare:
        compare_results(args.compare[0], args.compare[1])
    else:
        print("Usage:")
        print("  python save_results.py --list")
        print("  python save_results.py --compare validation_20260426_220000.json validation_20260426_230000.json")
