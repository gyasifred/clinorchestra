#!/usr/bin/env python3
"""
Direct function creation without requiring pandas
Creates function JSON files directly in functions directory
"""

import json
from pathlib import Path

# Medical functions to create
FUNCTIONS = [
    {
        "name": "calculate_bmi",
        "code": "def calculate_bmi(weight_kg, height_m):\n    '''Calculate BMI from weight and height'''\n    if height_m <= 0:\n        return None\n    bmi = weight_kg / (height_m ** 2)\n    return round(bmi, 2)",
        "description": "Calculate Body Mass Index from weight (kg) and height (m)",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "height_m": {"type": "number", "description": "Height in meters"}
        },
        "return_type": "number"
    },
    {
        "name": "kg_to_lbs",
        "code": "def kg_to_lbs(kg):\n    '''Convert kilograms to pounds'''\n    return round(kg * 2.20462, 2)",
        "description": "Convert weight from kilograms to pounds",
        "parameters": {
            "kg": {"type": "number", "description": "Weight in kilograms"}
        },
        "return_type": "number"
    },
    {
        "name": "lbs_to_kg",
        "code": "def lbs_to_kg(lbs):\n    '''Convert pounds to kilograms'''\n    return round(lbs / 2.20462, 2)",
        "description": "Convert weight from pounds to kilograms",
        "parameters": {
            "lbs": {"type": "number", "description": "Weight in pounds"}
        },
        "return_type": "number"
    },
    {
        "name": "cm_to_m",
        "code": "def cm_to_m(cm):\n    '''Convert centimeters to meters'''\n    return round(cm / 100, 3)",
        "description": "Convert height from centimeters to meters",
        "parameters": {
            "cm": {"type": "number", "description": "Height in centimeters"}
        },
        "return_type": "number"
    },
    {
        "name": "inches_to_cm",
        "code": "def inches_to_cm(inches):\n    '''Convert inches to centimeters'''\n    return round(inches * 2.54, 2)",
        "description": "Convert height from inches to centimeters",
        "parameters": {
            "inches": {"type": "number", "description": "Height in inches"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_weight_change_percent",
        "code": "def calculate_weight_change_percent(initial_weight, final_weight):\n    '''Calculate percentage weight change'''\n    if initial_weight == 0:\n        return None\n    change = ((final_weight - initial_weight) / initial_weight) * 100\n    return round(change, 2)",
        "description": "Calculate percentage weight change between two timepoints",
        "parameters": {
            "initial_weight": {"type": "number", "description": "Initial weight in any unit"},
            "final_weight": {"type": "number", "description": "Final weight in same unit"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_mean_arterial_pressure",
        "code": "def calculate_mean_arterial_pressure(systolic, diastolic):\n    '''Calculate MAP from BP readings'''\n    map_value = (systolic + 2 * diastolic) / 3\n    return round(map_value, 1)",
        "description": "Calculate Mean Arterial Pressure from systolic and diastolic BP",
        "parameters": {
            "systolic": {"type": "number", "description": "Systolic blood pressure (mmHg)"},
            "diastolic": {"type": "number", "description": "Diastolic blood pressure (mmHg)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_body_surface_area",
        "code": "def calculate_body_surface_area(weight_kg, height_cm):\n    '''Calculate BSA using Mosteller formula'''\n    import math\n    bsa = math.sqrt((weight_kg * height_cm) / 3600)\n    return round(bsa, 3)",
        "description": "Calculate Body Surface Area using Mosteller formula",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "height_cm": {"type": "number", "description": "Height in centimeters"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_ideal_body_weight",
        "code": "def calculate_ideal_body_weight(height_cm, sex):\n    '''Calculate IBW using Devine formula'''\n    height_inches = height_cm / 2.54\n    if sex.lower() in ['male', 'm']:\n        ibw_kg = 50 + 2.3 * (height_inches - 60)\n    elif sex.lower() in ['female', 'f']:\n        ibw_kg = 45.5 + 2.3 * (height_inches - 60)\n    else:\n        return None\n    return round(ibw_kg, 1)",
        "description": "Calculate Ideal Body Weight using Devine formula",
        "parameters": {
            "height_cm": {"type": "number", "description": "Height in centimeters"},
            "sex": {"type": "string", "description": "Sex: 'male' or 'female'"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_anion_gap",
        "code": "def calculate_anion_gap(sodium, chloride, bicarbonate):\n    '''Calculate serum anion gap'''\n    ag = sodium - (chloride + bicarbonate)\n    return round(ag, 1)",
        "description": "Calculate serum anion gap",
        "parameters": {
            "sodium": {"type": "number", "description": "Serum sodium (mEq/L)"},
            "chloride": {"type": "number", "description": "Serum chloride (mEq/L)"},
            "bicarbonate": {"type": "number", "description": "Serum bicarbonate (mEq/L)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_corrected_calcium",
        "code": "def calculate_corrected_calcium(calcium, albumin):\n    '''Calculate corrected calcium for hypoalbuminemia'''\n    corrected_ca = calcium + 0.8 * (4.0 - albumin)\n    return round(corrected_ca, 2)",
        "description": "Calculate corrected calcium for low albumin",
        "parameters": {
            "calcium": {"type": "number", "description": "Serum calcium (mg/dL)"},
            "albumin": {"type": "number", "description": "Serum albumin (g/dL)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_creatinine_clearance",
        "code": "def calculate_creatinine_clearance(age, weight_kg, creatinine, sex):\n    '''Calculate CrCl using Cockcroft-Gault'''\n    if sex.lower() in ['male', 'm']:\n        crcl = ((140 - age) * weight_kg) / (72 * creatinine)\n    elif sex.lower() in ['female', 'f']:\n        crcl = ((140 - age) * weight_kg) / (72 * creatinine) * 0.85\n    else:\n        return None\n    return round(crcl, 1)",
        "description": "Calculate creatinine clearance using Cockcroft-Gault equation",
        "parameters": {
            "age": {"type": "number", "description": "Age in years"},
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "creatinine": {"type": "number", "description": "Serum creatinine (mg/dL)"},
            "sex": {"type": "string", "description": "Sex: 'male' or 'female'"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_fluid_requirement",
        "code": "def calculate_fluid_requirement(weight_kg):\n    '''Calculate daily maintenance fluid requirement (Holliday-Segar)'''\n    if weight_kg <= 10:\n        fluid = weight_kg * 100\n    elif weight_kg <= 20:\n        fluid = 1000 + (weight_kg - 10) * 50\n    else:\n        fluid = 1500 + (weight_kg - 20) * 20\n    return round(fluid, 0)",
        "description": "Calculate daily maintenance fluid requirement using Holliday-Segar method",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_osmolality",
        "code": "def calculate_osmolality(sodium, glucose, bun):\n    '''Calculate serum osmolality'''\n    osm = 2 * sodium + (glucose / 18) + (bun / 2.8)\n    return round(osm, 1)",
        "description": "Calculate serum osmolality from electrolytes",
        "parameters": {
            "sodium": {"type": "number", "description": "Serum sodium (mEq/L)"},
            "glucose": {"type": "number", "description": "Serum glucose (mg/dL)"},
            "bun": {"type": "number", "description": "Blood urea nitrogen (mg/dL)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_pack_years",
        "code": "def calculate_pack_years(packs_per_day, years_smoked):\n    '''Calculate smoking pack-years'''\n    pack_years = packs_per_day * years_smoked\n    return round(pack_years, 1)",
        "description": "Calculate smoking history in pack-years",
        "parameters": {
            "packs_per_day": {"type": "number", "description": "Average packs smoked per day"},
            "years_smoked": {"type": "number", "description": "Number of years smoking"}
        },
        "return_type": "number"
    }
]

def main():
    print("Creating medical function files directly...")

    functions_dir = Path("functions")
    functions_dir.mkdir(exist_ok=True)

    created = 0
    for func in FUNCTIONS:
        # Create metadata JSON file
        json_file = functions_dir / f"function_{func['name']}.json"
        metadata = {
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"],
            "return_type": func["return_type"],
            "created_at": "2025-01-05T00:00:00"
        }

        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create Python code file
        py_file = functions_dir / f"function_{func['name']}.py"
        with open(py_file, 'w') as f:
            f.write(func["code"])

        created += 1
        print(f"✅ Created: {func['name']}")

    print(f"\n✅ Successfully created {created}/{len(FUNCTIONS)} functions")

if __name__ == "__main__":
    main()
