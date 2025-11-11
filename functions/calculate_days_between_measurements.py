from datetime import datetime
from dateutil import parser

def calculate_days_between_measurements(first_date: str, second_date: str) -> dict:
    """
    Calculate days between two measurement dates

    Args:
        first_date: First measurement date (various formats supported)
        second_date: Second measurement date (various formats supported)

    Returns:
        {
            'days_between': int,
            'weeks_between': float,
            'interpretation': str
        }
    """
    try:
        date1 = parser.parse(first_date)
        date2 = parser.parse(second_date)

        delta = abs((date2 - date1).days)
        weeks = delta / 7.0

        if delta == 0:
            interpretation = "Same day"
        elif delta == 1:
            interpretation = "1 day apart"
        elif delta < 7:
            interpretation = f"{delta} days apart (less than 1 week)"
        elif delta < 14:
            interpretation = f"{delta} days apart (approximately {weeks:.1f} weeks)"
        elif delta < 30:
            interpretation = f"{delta} days apart (approximately {weeks:.1f} weeks)"
        elif delta < 365:
            months = delta / 30.0
            interpretation = f"{delta} days apart (approximately {months:.1f} months)"
        else:
            years = delta / 365.0
            interpretation = f"{delta} days apart (approximately {years:.1f} years)"

        return {
            'days_between': delta,
            'weeks_between': round(weeks, 2),
            'interpretation': interpretation
        }

    except Exception as e:
        return {
            'days_between': None,
            'weeks_between': None,
            'interpretation': f"Error parsing dates: {str(e)}"
        }
