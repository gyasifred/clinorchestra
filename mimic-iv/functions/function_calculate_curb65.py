"""
Function: Calculate CURB-65 Score

CURB-65 is a clinical prediction rule for assessing severity of pneumonia
and determining whether hospitalization is needed.

Score components:
- C: Confusion (new onset)
- U: Urea (BUN) >19 mg/dL (>7 mmol/L)
- R: Respiratory rate ≥30/min
- B: Blood pressure (systolic <90 or diastolic ≤60 mmHg)
- 65: Age ≥65 years

Scoring range: 0-5 points
"""


def calculate_curb65(
    confusion: bool,
    bun: float,
    respiratory_rate: int,
    systolic_bp: int,
    diastolic_bp: int,
    age: int
) -> dict:
    """
    Calculate CURB-65 Score for pneumonia severity assessment

    Parameters:
    -----------
    confusion : bool
        New onset confusion or altered mental status
    bun : float
        Blood Urea Nitrogen in mg/dL
    respiratory_rate : int
        Respiratory rate in breaths per minute
    systolic_bp : int
        Systolic blood pressure in mmHg
    diastolic_bp : int
        Diastolic blood pressure in mmHg
    age : int
        Patient age in years

    Returns:
    --------
    dict with keys:
        - total_score: int (0-5)
        - confusion_point: int (0 or 1)
        - urea_point: int (0 or 1)
        - respiratory_point: int (0 or 1)
        - blood_pressure_point: int (0 or 1)
        - age_point: int (0 or 1)
        - severity: str (Low, Moderate, High)
        - recommendation: str
        - mortality_risk: str
    """

    score = 0
    points = {
        "confusion_point": 0,
        "urea_point": 0,
        "respiratory_point": 0,
        "blood_pressure_point": 0,
        "age_point": 0
    }

    # C: Confusion
    if confusion:
        points["confusion_point"] = 1
        score += 1

    # U: Urea (BUN > 19 mg/dL)
    if bun > 19:
        points["urea_point"] = 1
        score += 1

    # R: Respiratory rate ≥30
    if respiratory_rate >= 30:
        points["respiratory_point"] = 1
        score += 1

    # B: Blood pressure (systolic <90 OR diastolic ≤60)
    if systolic_bp < 90 or diastolic_bp <= 60:
        points["blood_pressure_point"] = 1
        score += 1

    # 65: Age ≥65
    if age >= 65:
        points["age_point"] = 1
        score += 1

    # Determine severity and recommendations
    if score <= 1:
        severity = "Low Risk"
        recommendation = "Consider outpatient treatment"
        mortality_risk = "<3%"
    elif score == 2:
        severity = "Moderate Risk"
        recommendation = "Consider short hospitalization or closely supervised outpatient treatment"
        mortality_risk = "9%"
    elif score == 3:
        severity = "High Risk"
        recommendation = "Hospital admission recommended"
        mortality_risk = "15-17%"
    else:  # score >= 4
        severity = "Severe"
        recommendation = "Urgent hospitalization, consider ICU admission"
        mortality_risk = ">20%"

    return {
        "total_score": score,
        **points,
        "severity": severity,
        "recommendation": recommendation,
        "mortality_risk": mortality_risk,
        "criteria_details": {
            "confusion": "Present" if confusion else "Absent",
            "bun": f"{bun} mg/dL {'(Elevated)' if bun > 19 else '(Normal)'}",
            "respiratory_rate": f"{respiratory_rate} /min {'(Elevated)' if respiratory_rate >= 30 else '(Normal)'}",
            "blood_pressure": f"{systolic_bp}/{diastolic_bp} mmHg {'(Hypotensive)' if (systolic_bp < 90 or diastolic_bp <= 60) else '(Normal)'}",
            "age": f"{age} years {'(≥65)' if age >= 65 else '(<65)'}"
        }
    }


def calculate_curb65_from_text(clinical_data: dict) -> dict:
    """
    Wrapper function to calculate CURB-65 from extracted clinical data

    This version is designed to work with data extracted from clinical notes

    Parameters:
    -----------
    clinical_data : dict
        Dictionary with keys: confusion, bun, respiratory_rate, systolic_bp, diastolic_bp, age

    Returns:
    --------
    Same as calculate_curb65()
    """
    return calculate_curb65(
        confusion=clinical_data.get("confusion", False),
        bun=clinical_data.get("bun", 0),
        respiratory_rate=clinical_data.get("respiratory_rate", 0),
        systolic_bp=clinical_data.get("systolic_bp", 120),
        diastolic_bp=clinical_data.get("diastolic_bp", 80),
        age=clinical_data.get("age", 0)
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Low risk patient
    print("="*60)
    print("Example 1: Low Risk Pneumonia")
    print("="*60)
    result1 = calculate_curb65(
        confusion=False,
        bun=15,
        respiratory_rate=18,
        systolic_bp=125,
        diastolic_bp=78,
        age=45
    )
    print(f"CURB-65 Score: {result1['total_score']}")
    print(f"Severity: {result1['severity']}")
    print(f"Recommendation: {result1['recommendation']}")
    print(f"Mortality Risk: {result1['mortality_risk']}")
    print()

    # Example 2: High risk patient
    print("="*60)
    print("Example 2: High Risk Pneumonia")
    print("="*60)
    result2 = calculate_curb65(
        confusion=True,
        bun=28,
        respiratory_rate=32,
        systolic_bp=85,
        diastolic_bp=55,
        age=72
    )
    print(f"CURB-65 Score: {result2['total_score']}")
    print(f"Severity: {result2['severity']}")
    print(f"Recommendation: {result2['recommendation']}")
    print(f"Mortality Risk: {result2['mortality_risk']}")
    print("\nCriteria Details:")
    for criterion, value in result2['criteria_details'].items():
        print(f"  {criterion}: {value}")
