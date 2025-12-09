# -*- coding: utf-8 -*-
"""
Tinvest Analytics Modules
Módulos para el dashboard de análisis
"""

__version__ = "1.0.0"
__author__ = "Tinvest Analytics Team"

# Importar módulos principales
from . import data_loader
from . import eda
from . import cohort_analysis
from . import rfm_segmentation
from . import churn_prediction
from . import forecasting
from . import executive_summary

__all__ = [
    'data_loader',
    'eda',
    'cohort_analysis',
    'rfm_segmentation',
    'churn_prediction',
    'forecasting',
    'executive_summary'
]