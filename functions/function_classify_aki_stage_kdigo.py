def classify_aki_stage_kdigo(baseline_cr: float, current_cr: float,
                             cr_change_48h: float = None,
                             on_rrt: bool = False) -> dict:
    """
    Classify AKI stage per KDIGO criteria

    KDIGO AKI Staging:
    - Stage 1: Cr increase ≥0.3 mg/dL within 48h OR 1.5-1.9x baseline
    - Stage 2: Cr increase 2.0-2.9x baseline
    - Stage 3: Cr increase ≥3.0x baseline OR Cr ≥4.0 mg/dL OR initiation of RRT

    Args:
        baseline_cr: Baseline creatinine (mg/dL)
        current_cr: Current creatinine (mg/dL)
        cr_change_48h: Change in Cr over 48h (mg/dL)
        on_rrt: Patient on renal replacement therapy

    Returns:
        {
            'aki_stage': 0-3,
            'criteria_met': str,
            'cr_ratio': float,
            'absolute_change': float
        }
    """
    cr_ratio = current_cr / baseline_cr if baseline_cr > 0 else 0
    absolute_change = current_cr - baseline_cr

    # Stage 3
    if on_rrt or current_cr >= 4.0 or cr_ratio >= 3.0:
        return {
            'aki_stage': 3,
            'criteria_met': f"Stage 3: RRT={on_rrt}, Cr={current_cr:.1f}, Ratio={cr_ratio:.2f}x",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # Stage 2
    if cr_ratio >= 2.0:
        return {
            'aki_stage': 2,
            'criteria_met': f"Stage 2: Ratio {cr_ratio:.2f}x baseline",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # Stage 1
    if cr_ratio >= 1.5 or (cr_change_48h is not None and cr_change_48h >= 0.3):
        criteria = []
        if cr_ratio >= 1.5:
            criteria.append(f"{cr_ratio:.2f}x baseline")
        if cr_change_48h and cr_change_48h >= 0.3:
            criteria.append(f"+{cr_change_48h:.1f} mg/dL in 48h")

        return {
            'aki_stage': 1,
            'criteria_met': f"Stage 1: {', '.join(criteria)}",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # No AKI
    return {
        'aki_stage': 0,
        'criteria_met': "No AKI criteria met",
        'cr_ratio': cr_ratio,
        'absolute_change': absolute_change
    }
