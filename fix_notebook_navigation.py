#!/usr/bin/env python3
"""Fix navigation links in the notebook"""

import json
from pathlib import Path

notebook_path = Path('course_site/day3/notebooks/Day3_Session2_Flood_Mapping_UNet.ipynb')

# Read notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Fix navigation links in the last markdown cell
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Fix session links
        if 'session1.qmd' in source:
            source = source.replace(
                '[← Back to Session 1](session1.qmd)',
                '[← Back to Session 1](../sessions/session1.qmd)'
            )
        if 'session3.qmd' in source:
            source = source.replace(
                '[Next: Session 3 - Object Detection →](session3.qmd)',
                '[Next: Session 3 - Object Detection →](../sessions/session3.qmd)'
            )
            source = source.replace(
                '[Preview Session 3 →](session3.qmd)',
                '[Preview Session 3 →](../sessions/session3.qmd)'
            )
        
        cell['source'] = [source]

# Write back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✓ Updated navigation links in notebook")
