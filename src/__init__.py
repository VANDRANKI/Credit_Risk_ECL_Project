"""
CECL Credit Risk Modeling Package
=================================

A professional credit risk modeling toolkit for CECL-compliant
Expected Credit Loss estimation.

Modules:
--------
- data_processing: Data cleaning and transformation utilities
- feature_engineering: Feature creation and encoding functions
- modeling: PD model training and prediction utilities
- ecl_calculator: ECL computation functions
- stress_testing: Stress scenario application and analysis
- visualization: Professional chart generation utilities

Author: Prabhu
"""

from . import data_processing
from . import feature_engineering
from . import modeling
from . import ecl_calculator
from . import stress_testing
from . import visualization

__version__ = "1.0.0"
__author__ = "Prabhu"

__all__ = [
    "data_processing",
    "feature_engineering",
    "modeling",
    "ecl_calculator",
    "stress_testing",
    "visualization"
]
