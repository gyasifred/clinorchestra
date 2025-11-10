"""
Function: Calculate SOFA Score (Sequential Organ Failure Assessment)

The SOFA score is used to assess organ dysfunction in critically ill patients,
particularly in sepsis cases.

Scoring range: 0-24 (higher scores indicate worse organ dysfunction)
"""


def calculate_sofa_score(
    pao2_fio2_ratio: float = None,
    platelets: float = None,
    bilirubin: float = None,
    map_or_vasopressor: dict = None,  # {"map": value} or {"vasopressor": "drug_name"}
    gcs: int = None,
    creatinine: float = None,
    urine_output: float = None
) -> dict:
    """
    Calculate SOFA (Sequential Organ Failure Assessment) Score

    Parameters:
    -----------
    pao2_fio2_ratio : float
        PaO2/FiO2 ratio in mmHg
    platelets : float
        Platelet count in 10^3/μL
    bilirubin : float
        Total bilirubin in mg/dL
    map_or_vasopressor : dict
        Either {"map": value} for mean arterial pressure in mmHg
        or {"vasopressor": "drug_name"} if on vasopressors
    gcs : int
        Glasgow Coma Scale score (3-15)
    creatinine : float
        Serum creatinine in mg/dL
    urine_output : float
        Urine output in mL/day

    Returns:
    --------
    dict with keys:
        - total_score: int (0-24)
        - respiration_score: int (0-4)
        - coagulation_score: int (0-4)
        - liver_score: int (0-4)
        - cardiovascular_score: int (0-4)
        - cns_score: int (0-4)
        - renal_score: int (0-4)
        - interpretation: str
        - mortality_risk: str
    """

    scores = {
        "respiration_score": 0,
        "coagulation_score": 0,
        "liver_score": 0,
        "cardiovascular_score": 0,
        "cns_score": 0,
        "renal_score": 0
    }

    # Respiration (PaO2/FiO2)
    if pao2_fio2_ratio is not None:
        if pao2_fio2_ratio < 100:
            scores["respiration_score"] = 4
        elif pao2_fio2_ratio < 200:
            scores["respiration_score"] = 3
        elif pao2_fio2_ratio < 300:
            scores["respiration_score"] = 2
        elif pao2_fio2_ratio < 400:
            scores["respiration_score"] = 1

    # Coagulation (Platelets)
    if platelets is not None:
        if platelets < 20:
            scores["coagulation_score"] = 4
        elif platelets < 50:
            scores["coagulation_score"] = 3
        elif platelets < 100:
            scores["coagulation_score"] = 2
        elif platelets < 150:
            scores["coagulation_score"] = 1

    # Liver (Bilirubin)
    if bilirubin is not None:
        if bilirubin >= 12.0:
            scores["liver_score"] = 4
        elif bilirubin >= 6.0:
            scores["liver_score"] = 3
        elif bilirubin >= 2.0:
            scores["liver_score"] = 2
        elif bilirubin >= 1.2:
            scores["liver_score"] = 1

    # Cardiovascular (MAP or vasopressors)
    if map_or_vasopressor is not None:
        if "vasopressor" in map_or_vasopressor:
            vasopressor = map_or_vasopressor["vasopressor"].lower()
            if any(drug in vasopressor for drug in ["epinephrine", "norepinephrine"]):
                scores["cardiovascular_score"] = 4
            elif any(drug in vasopressor for drug in ["dopamine", "dobutamine"]):
                scores["cardiovascular_score"] = 3
            else:
                scores["cardiovascular_score"] = 2
        elif "map" in map_or_vasopressor:
            map_value = map_or_vasopressor["map"]
            if map_value < 70:
                scores["cardiovascular_score"] = 1

    # CNS (Glasgow Coma Scale)
    if gcs is not None:
        if gcs < 6:
            scores["cns_score"] = 4
        elif gcs < 10:
            scores["cns_score"] = 3
        elif gcs < 13:
            scores["cns_score"] = 2
        elif gcs < 15:
            scores["cns_score"] = 1

    # Renal (Creatinine or Urine Output)
    if creatinine is not None:
        if creatinine >= 5.0:
            scores["renal_score"] = 4
        elif creatinine >= 3.5:
            scores["renal_score"] = 3
        elif creatinine >= 2.0:
            scores["renal_score"] = 2
        elif creatinine >= 1.2:
            scores["renal_score"] = 1

    if urine_output is not None and urine_output < 500:
        scores["renal_score"] = max(scores["renal_score"], 3)
        if urine_output < 200:
            scores["renal_score"] = 4

    # Calculate total
    total = sum(scores.values())

    # Interpretation
    if total >= 15:
        interpretation = "Severe organ dysfunction"
        mortality_risk = "Very High (>90%)"
    elif total >= 12:
        interpretation = "Severe organ dysfunction"
        mortality_risk = "High (40-50%)"
    elif total >= 8:
        interpretation = "Moderate organ dysfunction"
        mortality_risk = "Moderate (15-20%)"
    elif total >= 4:
        interpretation = "Mild organ dysfunction"
        mortality_risk = "Low (5-10%)"
    else:
        interpretation = "Minimal organ dysfunction"
        mortality_risk = "Very Low (<5%)"

    return {
        "total_score": total,
        **scores,
        "interpretation": interpretation,
        "mortality_risk": mortality_risk
    }


# Example usage for ClinOrchestra
if __name__ == "__main__":
    # Example: Septic patient
    result = calculate_sofa_score(
        pao2_fio2_ratio=180,  # Respiratory failure
        platelets=95,         # Thrombocytopenia
        bilirubin=2.5,        # Elevated bilirubin
        map_or_vasopressor={"vasopressor": "norepinephrine"},  # On pressors
        gcs=12,               # Mild altered mental status
        creatinine=2.8        # Acute kidney injury
    )

    print("SOFA Score Calculation:")
    print(f"Total Score: {result['total_score']}")
    print(f"  - Respiration: {result['respiration_score']}")
    print(f"  - Coagulation: {result['coagulation_score']}")
    print(f"  - Liver: {result['liver_score']}")
    print(f"  - Cardiovascular: {result['cardiovascular_score']}")
    print(f"  - CNS: {result['cns_score']}")
    print(f"  - Renal: {result['renal_score']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Mortality Risk: {result['mortality_risk']}")
