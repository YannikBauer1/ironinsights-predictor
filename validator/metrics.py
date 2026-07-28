"""
Validation/testing module - compare predictions to actual results.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ActualResult:
    """Actual competition results."""
    athlete_id: str
    name: str
    actual_place: int


@dataclass
class ValidationMetrics:
    """Accuracy metrics for a prediction."""
    event_id: str
    division: str
    num_athletes: int
    
    # Exact match predictions
    exact_matches: int
    exact_match_rate: float
    
    # Winner accuracy (predicted #1 actually won)
    winner_correct: bool
    winner_accuracy: float
    
    # Top 3 accuracy
    top3_correct: int
    top3_accuracy: float
    
    # Top 5 accuracy
    top5_correct: int
    top5_accuracy: float
    
    # Top 10 accuracy
    top10_correct: int
    top10_accuracy: float
    
    # Mean Absolute Error (average place difference)
    mae: float
    
    # Detailed comparison
    comparisons: List[Dict]

    # Model P(1st) for the athlete ranked #1 in this division (0–1)
    predicted_winner_win_probability: float = 0.0

    # Optional metadata for mismatch investigation
    division_id: str = ""
    event_name: str = ""


def compare_prediction_to_actual(
    predicted: List[Dict],
    actual: List[ActualResult]
) -> ValidationMetrics:
    """
    Compare predicted vs actual results.
    
    Args:
        predicted: List of predictions with athlete_id, predicted_place
        actual: List of actual results
    
    Returns:
        ValidationMetrics with accuracy stats
    """
    # Create lookup dictionaries
    pred_dict = {p['athlete_id']: p for p in predicted}
    actual_dict = {a.athlete_id: a for a in actual}
    
    exact_matches = 0
    winner_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_error = 0
    comparisons = []
    
    for athlete_id in pred_dict:
        pred_place = pred_dict[athlete_id]['predicted_place']
        
        if athlete_id in actual_dict:
            actual_place = actual_dict[athlete_id].actual_place
            
            # Check exact match
            if pred_place == actual_place:
                exact_matches += 1
            
            # Check winner (predicted #1 actually won)
            if pred_place == 1 and actual_place == 1:
                winner_correct = 1
            
            # Check top 3 (both in top 3)
            if pred_place <= 3 and actual_place <= 3:
                top3_correct += 1
            
            # Check top 5 (both in top 5)
            if pred_place <= 5 and actual_place <= 5:
                top5_correct += 1
            
            # Check top 10 (both in top 10)
            if pred_place <= 10 and actual_place <= 10:
                top10_correct += 1
            
            # Calculate error
            error = abs(pred_place - actual_place)
            total_error += error
            
            comparisons.append({
                'athlete_id': athlete_id,
                'name': actual_dict[athlete_id].name,
                'predicted': pred_place,
                'actual': actual_place,
                'error': error,
                'exact_match': pred_place == actual_place
            })
    
    num_athletes = len(predicted)
    
    return ValidationMetrics(
        event_id="",  # Set by caller
        division="",
        num_athletes=num_athletes,
        exact_matches=exact_matches,
        exact_match_rate=exact_matches / num_athletes if num_athletes > 0 else 0,
        winner_correct=winner_correct == 1,
        winner_accuracy=1.0 if winner_correct == 1 else 0.0,
        top3_correct=top3_correct,
        top3_accuracy=top3_correct / min(3, num_athletes) if num_athletes > 0 else 0,
        top5_correct=top5_correct,
        top5_accuracy=top5_correct / min(5, num_athletes) if num_athletes > 0 else 0,
        top10_correct=top10_correct,
        top10_accuracy=top10_correct / min(10, num_athletes) if num_athletes > 0 else 0,
        mae=total_error / num_athletes if num_athletes > 0 else 0,
        comparisons=comparisons
    )


def print_validation_report(metrics: ValidationMetrics):
    """Print a formatted validation report."""
    print("=" * 60)
    print(f"VALIDATION REPORT: {metrics.event_id}")
    print(f"Division: {metrics.division}")
    print(f"Athletes: {metrics.num_athletes}")
    print("=" * 60)
    print()
    
    print(f"Exact Match Rate:     {metrics.exact_match_rate:.1%} ({metrics.exact_matches}/{metrics.num_athletes})")
    print(f"Top 3 Accuracy:       {metrics.top3_accuracy:.1%} ({metrics.top3_correct}/3)")
    print(f"Mean Absolute Error:  {metrics.mae:.1f} places")
    print()
    
    print("Detailed Comparisons:")
    print("-" * 60)
    print(f"{'Athlete':<25} {'Predicted':<12} {'Actual':<10} {'Error':<8}")
    print("-" * 60)
    
    for comp in sorted(metrics.comparisons, key=lambda x: x['predicted']):
        name = comp['name'][:24]
        mark = "✓" if comp['exact_match'] else "✗"
        print(f"{name:<25} {comp['predicted']:<12} {comp['actual']:<10} {comp['error']:<8} {mark}")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    # Example validation
    example_predicted = [
        {'athlete_id': '1', 'predicted_place': 1},
        {'athlete_id': '2', 'predicted_place': 2},
        {'athlete_id': '3', 'predicted_place': 3},
    ]
    
    example_actual = [
        ActualResult('1', 'Carlos', 1),  # ✓ Correct
        ActualResult('2', 'Ashley', 3),  # ✗ Predicted 2, got 3
        ActualResult('3', 'Mike', 2),     # ✗ Predicted 3, got 2
    ]
    
    metrics = compare_prediction_to_actual(example_predicted, example_actual)
    metrics.event_id = "Tri-City Classic 2026"
    metrics.division = "Men's Physique"
    
    print_validation_report(metrics)
