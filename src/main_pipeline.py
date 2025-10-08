#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
===========================================================
 Script Name:    main_pipeline.py
 Author:         David Buchner, Imperial College London
 Created:        08/10/2025
 Last Modified:  08/10/2025
 Version:        -
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
'''
# -----------------------------
# Load Python packages
import numpy as np

# -----------------------------