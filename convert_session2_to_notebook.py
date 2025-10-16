#!/usr/bin/env python3
"""
Convert Session 2 QMD to Jupyter Notebook
Preserves all content, code blocks, and formatting
"""

import json
import re
from pathlib import Path

def create_notebook_cell(cell_type, content, metadata=None):
    """Create a notebook cell"""
    cell = {
        'cell_type': cell_type,
        'metadata': metadata or {},
        'source': content if isinstance(content, list) else [content]
    }
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    return cell

def parse_qmd_to_notebook(qmd_path, output_path):
    """Parse QMD file and convert to Jupyter notebook"""
    
    with open(qmd_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by code blocks
    parts = re.split(r'(```(?:python|javascript|bash)\n.*?\n```)', content, flags=re.DOTALL)
    
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3 (ipykernel)',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.9.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }
    
    # Add YAML header as raw cell
    yaml_header = '''---
title: "Session 2: Hands-on Flood Mapping with U-Net and Sentinel-1 SAR"
subtitle: "Practical Implementation for Disaster Risk Reduction"
format:
  html:
    code-fold: false
    code-tools: true
    toc: true
    toc-depth: 3
    number-sections: false
date: last-modified
author: "CopPhil Advanced Training Program"
execute:
  eval: false
  echo: true
jupyter: python3
---'''
    
    notebook['cells'].append(create_notebook_cell('raw', yaml_header))
    
    # Process content
    current_markdown = []
    
    for i, part in enumerate(parts):
        if part.strip() == '':
            continue
            
        # Check if it's a code block
        code_match = re.match(r'```(python|javascript|bash)\n(.*?)\n```', part, re.DOTALL)
        
        if code_match:
            # Save accumulated markdown
            if current_markdown:
                md_text = ''.join(current_markdown).strip()
                if md_text:
                    # Skip YAML header if present
                    if not md_text.startswith('---\ntitle:'):
                        notebook['cells'].append(create_notebook_cell('markdown', md_text))
                current_markdown = []
            
            # Add code cell
            lang = code_match.group(1)
            code = code_match.group(2)
            
            if lang == 'python':
                # Clean up code
                code = code.strip()
                if code:
                    notebook['cells'].append(create_notebook_cell('code', code))
            elif lang in ['bash', 'javascript']:
                # Add as markdown code block for non-Python
                notebook['cells'].append(create_notebook_cell('markdown', part))
        else:
            # Accumulate markdown
            current_markdown.append(part)
    
    # Add remaining markdown
    if current_markdown:
        md_text = ''.join(current_markdown).strip()
        if md_text and not md_text.startswith('---\ntitle:'):
            notebook['cells'].append(create_notebook_cell('markdown', md_text))
    
    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Notebook created: {output_path}")
    print(f"  Total cells: {len(notebook['cells'])}")
    
    # Count cell types
    markdown_cells = sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')
    code_cells = sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')
    print(f"  Markdown cells: {markdown_cells}")
    print(f"  Code cells: {code_cells}")

if __name__ == '__main__':
    qmd_file = Path('course_site/day3/sessions/session2.qmd')
    notebook_file = Path('course_site/day3/notebooks/Day3_Session2_Flood_Mapping_UNet.ipynb')
    
    # Create notebooks directory if needed
    notebook_file.parent.mkdir(parents=True, exist_ok=True)
    
    parse_qmd_to_notebook(qmd_file, notebook_file)
    
    print(f"\n✓ Conversion complete!")
    print(f"\nTo render with Quarto:")
    print(f"  quarto render {notebook_file}")
