#!/usr/bin/env python3
"""
Add PDF format option to all session QMD files
"""

import os
import re
from pathlib import Path

def add_pdf_format(qmd_file):
    """Add PDF format to a QMD file's YAML front matter"""

    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has format section
    if 'format:' in content:
        print(f"  ⏭ Skipped {qmd_file.name} (already has format section)")
        return False

    # Split into YAML front matter and content
    parts = content.split('---', 2)

    if len(parts) < 3:
        print(f"  ⚠ Warning: {qmd_file.name} doesn't have proper YAML front matter")
        return False

    yaml_content = parts[1]
    main_content = parts[2]

    # Add format section to YAML
    format_addition = """format:
  html:
    toc: true
    toc-depth: 3
    code-fold: false
  pdf:
    toc: true
    toc-depth: 3
    number-sections: true
    colorlinks: true
    geometry:
      - margin=1in
    include-in-header:
      text: |
        \\usepackage{fancyhdr}
        \\pagestyle{fancy}
        \\fancyhead[L]{CopPhil Training}
        \\fancyhead[R]{\\thepage}
"""

    # Combine back together
    new_content = f"---{yaml_content}{format_addition}---{main_content}"

    # Write back
    with open(qmd_file, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"  ✓ Added PDF format to {qmd_file.name}")
    return True

def add_pdf_download_button(qmd_file):
    """Add PDF download button after Session Overview section"""

    with open(qmd_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has PDF download button
    if 'Download PDF' in content or 'download-pdf' in content:
        return False

    # Find the Session Overview section and add button after it
    pattern = r'(## Session Overview\n\n.*?\n\n)'

    pdf_filename = qmd_file.stem + '.pdf'
    pdf_button = f'''
::: {{.callout-note icon=false}}
## 📄 Download Session Materials
This session is also available as a PDF for offline reading and printing.

[📥 Download PDF]({pdf_filename}){{.btn .btn-primary target="_blank"}}
:::

---

'''

    # Try to insert after Session Overview
    if '## Session Overview' in content:
        new_content = re.sub(
            pattern,
            r'\1' + pdf_button,
            content,
            count=1,
            flags=re.DOTALL
        )

        if new_content != content:
            with open(qmd_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  ✓ Added PDF download button to {qmd_file.name}")
            return True

    return False

def main():
    base_dir = Path('course_site')

    # Find all session QMD files
    session_files = []
    for day_dir in ['day1', 'day2', 'day3', 'day4']:
        sessions_dir = base_dir / day_dir / 'sessions'
        if sessions_dir.exists():
            session_files.extend(sessions_dir.glob('session*.qmd'))

    print("Adding PDF format to session pages...")
    print("="*60)

    modified_count = 0
    button_count = 0

    for qmd_file in sorted(session_files):
        print(f"\nProcessing: {qmd_file.relative_to(base_dir)}")

        # Add PDF format to YAML
        if add_pdf_format(qmd_file):
            modified_count += 1

        # Add download button
        if add_pdf_download_button(qmd_file):
            button_count += 1

    print("\n" + "="*60)
    print(f"✓ Modified {modified_count} file(s) with PDF format")
    print(f"✓ Added {button_count} PDF download button(s)")
    print(f"✓ Total session files processed: {len(session_files)}")
    print("\nNext steps:")
    print("  1. Run 'quarto render' to generate PDFs")
    print("  2. PDFs will be created alongside HTML files")
    print("  3. Download buttons will link to these PDFs")

if __name__ == '__main__':
    main()
