#!/usr/bin/env python3
"""Fix notebook YAML to disable execution properly"""

import json
from pathlib import Path

notebook_path = Path('course_site/day3/notebooks/Day3_Session2_Flood_Mapping_UNet.ipynb')

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Update YAML in first raw cell
yaml_header = '''---
title: "Session 2: Hands-on Flood Mapping with U-Net and Sentinel-1 SAR"
subtitle: "Practical Implementation for Disaster Risk Reduction"
format:
  html:
    code-fold: show
    code-tools: true
    toc: true
    toc-depth: 3
    number-sections: false
    css: ../../styles/custom.css
date: last-modified
author: "CopPhil Advanced Training Program"
execute:
  enabled: false
  eval: false
  echo: true
jupyter: python3
---'''

# Find and update raw cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'raw':
        cell['source'] = [yaml_header]
        break

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Updated notebook YAML: {notebook_path}")
