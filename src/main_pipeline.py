#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================
 Script Name:    main_pipeline.py
 Author:         David Buchner, Imperial College London
 Created:        08/10/2025
 Last Modified:  08/10/2025
 Description:
    Script to execute the image segmentation pipeline.
    The pipline consists of:
    1. Loading and preprocessing data
    2. Threshholding and watershed segmentation
    3. Segmentation labelling
    4. Visualization segmentation

 Usage:
    python main_pipeline.py /path/to/data/

 Requirements:
    - Python 3.x
    - Required libraries:
        * numpy

 Notes:

===========================================================
"""

# -----------------------------
# Load Python packages
import numpy as np
from load_dataset import load_tif_sequence

# -----------------------------
# 1. Load the 3D image/dataset onto a numpy array
# i.e. into a format that allows efficient manipulation in the later steps
stack = load_tif_sequence("../imaging_data/spherical_particles", start=100, end=199)

# -----------------------------