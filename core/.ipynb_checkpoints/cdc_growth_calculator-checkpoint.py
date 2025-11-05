"""
CDC Growth Calculator Module

Calculates z-scores and percentiles using CDC growth chart data for children and adolescents aged 2-20 years (and head circumference for 0-36 months).
Uses the LMS method (Lambda-Mu-Sigma).

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: HeiderLab
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
import math
from scipy import stats
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Sex(Enum):
    """Sex enumeration following CDC coding standards"""
    MALE = 1
    FEMALE = 2

class GrowthMetric(Enum):
    """Available growth metrics"""
    WEIGHT_FOR_AGE = "weight_for_age"
    STATURE_FOR_AGE = "stature_for_age"
    WEIGHT_FOR_STATURE = "weight_for_stature"
    BMI_FOR_AGE = "bmi_for_age"
    HEAD_CIRCUMFERENCE = "head_circumference"

@dataclass
class GrowthResult:
    """Result of a growth calculation"""
    percentile: float
    z_score: float
    reference_value: float
    metric: str
    age_months: Optional[float] = None
    height_cm: Optional[float] = None

class CDCGrowthCalculator:
    """
    CDC Growth Calculator using LMS method for percentile and z-score calculations
    """
    
    def __init__(self, data_directory: str = "cdc_data"):
        """
        Initialize the calculator with CDC data
        
        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_directory = Path(data_directory)
        self.data = {}
        self._load_growth_data()
    
    def _load_growth_data(self):
        """Load all CDC growth chart data files"""
        file_mappings = {
            GrowthMetric.WEIGHT_FOR_AGE: "wtage.csv",
            GrowthMetric.STATURE_FOR_AGE: "statage.csv",
            GrowthMetric.WEIGHT_FOR_STATURE: "wtstat.csv",
            GrowthMetric.BMI_FOR_AGE: "bmiagerev.csv",
            GrowthMetric.HEAD_CIRCUMFERENCE: "hcageinf.csv"
        }
        
        for metric, filename in file_mappings.items():
            filepath = self.data_directory / filename
            try:
                df = pd.read_csv(filepath)
                
                # Handle different column names for weight-for-stature
                if metric == GrowthMetric.WEIGHT_FOR_STATURE:
                    if 'Height' in df.columns and 'Agemos' not in df.columns:
                        df = df.rename(columns={'Height': 'Agemos'})
                
                # Clean up BMI data
                if metric == GrowthMetric.BMI_FOR_AGE:
                    df = df[df['Sex'].isin(['1', '2', 1, 2])].copy()
                    for col in ['Sex', 'Agemos', 'L', 'M', 'S']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    df = df.dropna(subset=['Sex', 'Agemos', 'L', 'M', 'S'])
                
                # Ensure correct data types
                required_cols = ['Sex', 'Agemos', 'L', 'M', 'S']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.error(f"Missing required columns in {filename}: {missing_cols}")
                    self.data[metric] = pd.DataFrame()
                    continue
                
                df['Sex'] = pd.to_numeric(df['Sex'], errors='coerce').astype('Int64')
                df['Agemos'] = pd.to_numeric(df['Agemos'], errors='coerce')
                df['L'] = pd.to_numeric(df['L'], errors='coerce')
                df['M'] = pd.to_numeric(df['M'], errors='coerce')
                df['S'] = pd.to_numeric(df['S'], errors='coerce')
                
                df = df.dropna(subset=['Sex', 'Agemos', 'L', 'M', 'S'])
                df['Sex'] = df['Sex'].astype(int)
                
                self.data[metric] = df
                logger.info(f"Loaded {len(df)} data points for {metric.value}")
                
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                self.data[metric] = pd.DataFrame()
    
    def _get_lms_parameters(self, metric: GrowthMetric, sex: Sex, 
                           age_months: float, height_cm: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
        """
        Get LMS parameters for specific metric, sex, and age
        
        Returns:
            Tuple of (L, M, S) parameters or None if not found
        """
        if metric not in self.data or self.data[metric].empty:
            return None
        
        df = self.data[metric]
        sex_data = df[df['Sex'] == sex.value].copy()
        if sex_data.empty:
            return None
        
        if metric == GrowthMetric.WEIGHT_FOR_STATURE:
            if height_cm is None:
                return None
            closest_idx = (sex_data['Agemos'] - height_cm).abs().idxmin()
            row = sex_data.loc[closest_idx]
        else:
            closest_idx = (sex_data['Agemos'] - age_months).abs().idxmin()
            row = sex_data.loc[closest_idx]
        
        return row['L'], row['M'], row['S']
    
    def calculate_growth_percentile(self, 
                                   metric: GrowthMetric, 
                                   sex: Sex, 
                                   value: float,
                                   age_months: Optional[float] = None,
                                   height_cm: Optional[float] = None) -> Optional[GrowthResult]:
        """
        Calculate growth percentile and z-score using LMS method
        
        Args:
            metric: Growth metric to calculate
            sex: Patient sex
            value: Measured value
            age_months: Age in months (for age-based metrics)
            height_cm: Height in cm (for weight-for-stature)
            
        Returns:
            GrowthResult object or None if calculation fails
        """
        try:
            lms = self._get_lms_parameters(metric, sex, age_months, height_cm)
            if lms is None:
                return None
            
            L, M, S = lms
            
            if L != 0:
                z_score = (((value / M) ** L) - 1) / (L * S)
            else:
                z_score = math.log(value / M) / S
            
            percentile = stats.norm.cdf(z_score) * 100
            
            return GrowthResult(
                percentile=round(percentile, 2),
                z_score=round(z_score, 2),
                reference_value=M,
                metric=metric.value,
                age_months=age_months,
                height_cm=height_cm
            )
            
        except Exception as e:
            logger.error(f"Error calculating percentile for {metric.value}: {e}")
            return None
    
    def calculate_value_for_percentile(self, 
                                      metric: GrowthMetric, 
                                      sex: Sex, 
                                      percentile: float,
                                      age_months: Optional[float] = None,
                                      height_cm: Optional[float] = None) -> Optional[float]:
        """
        Calculate the value corresponding to a specific percentile
        
        Args:
            metric: Growth metric
            sex: Sex
            percentile: Desired percentile (0-100)
            age_months: Age in months (for age-based metrics)
            height_cm: Height in cm (for weight-for-stature)
            
        Returns:
            Value at the specified percentile
        """
        try:
            lms = self._get_lms_parameters(metric, sex, age_months, height_cm)
            if lms is None:
                return None
            
            L, M, S = lms
            z_score = stats.norm.ppf(percentile / 100)
            
            if L != 0:
                value = M * ((L * S * z_score + 1) ** (1/L))
            else:
                value = M * math.exp(S * z_score)
                
            return round(value, 2)
            
        except Exception as e:
            logger.error(f"Error calculating value for percentile: {e}")
            return None
    
    def batch_calculate(self, 
                       measurements: List[Dict[str, any]], 
                       metric: GrowthMetric) -> List[Optional[GrowthResult]]:
        """
        Calculate percentiles for multiple measurements
        
        Args:
            measurements: List of dicts with keys: sex, value, age_months (and height_cm if needed)
            metric: Growth metric to calculate
            
        Returns:
            List of GrowthResult objects
        """
        results = []
        for measurement in measurements:
            try:
                result = self.calculate_growth_percentile(
                    metric=metric,
                    sex=Sex(measurement['sex']),
                    value=measurement['value'],
                    age_months=measurement.get('age_months'),
                    height_cm=measurement.get('height_cm')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing measurement {measurement}: {e}")
                results.append(None)
        
        return results
