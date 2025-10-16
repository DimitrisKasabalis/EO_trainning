#!/usr/bin/env python3
"""
Standardize all session files to have the same structure:
1. Learning Objectives
2. Presentation Slides (iframe)
3. PDF Download Box
"""

import re
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Pattern for all sessions
SESSIONS = [
    ("day1", [1, 2, 3, 4]),
    ("day2", [1, 2, 3, 4]),
    ("day3", [1, 2, 3, 4]),
    ("day4", [1, 2, 3, 4]),
]


def standardize_session(filepath):
    """Standardize a single session file"""
    print(f"Processing: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if PDF download box already exists
    pdf_box_pattern = r'::: \{\.callout-note icon=false\}\s+## 📄 Download Session Materials.*?:::'

    # Extract the PDF download box if it exists
    pdf_box_match = re.search(pdf_box_pattern, content, re.DOTALL)

    if pdf_box_match:
        # Remove the existing PDF box from wherever it is
        content_no_pdf = re.sub(pdf_box_pattern, '', content, flags=re.DOTALL)
    else:
        content_no_pdf = content
        # Create a new PDF box
        session_num = os.path.basename(filepath).replace('session', '').replace('.qmd', '')
        pdf_box_match = type('obj', (object,), {
            'group': lambda self, n=0: f'''::: {{.callout-note icon=false}}
## 📄 Download Session Materials
This session is also available as a PDF for offline reading and printing.

[📥 Download PDF](../pdfs/session{session_num}.pdf){{.btn .btn-primary target="_blank"}}
:::'''
        })()

    # Find the presentation iframe section
    iframe_pattern = r'(## Presentation Slides.*?<iframe.*?</iframe>.*?:::)'
    iframe_match = re.search(iframe_pattern, content_no_pdf, re.DOTALL)

    if not iframe_match:
        print(f"  ⚠️  No iframe found in {filepath}, skipping...")
        return False

    # Get the iframe section
    iframe_section = iframe_match.group(1)

    # Insert PDF box after the iframe section
    new_iframe_section = iframe_section + '\n\n' + pdf_box_match.group(0)

    # Replace the old iframe section with the new one (including PDF box)
    final_content = content_no_pdf.replace(iframe_section, new_iframe_section)

    # Clean up any duplicate separators
    final_content = re.sub(r'\n---\n\n---\n', '\n---\n', final_content)
    final_content = re.sub(r'\n\n\n+', '\n\n', final_content)

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"  ✓ Updated successfully")
    return True


def main():
    """Process all session files"""
    updated_count = 0
    skipped_count = 0

    for day, sessions in SESSIONS:
        for session_num in sessions:
            filepath = BASE_DIR / day / "sessions" / f"session{session_num}.qmd"

            # Skip day1/session1 as we already updated it manually
            if str(filepath).endswith("day1/sessions/session1.qmd"):
                print(f"Skipping {filepath} (already updated manually)")
                continue

            if filepath.exists():
                success = standardize_session(filepath)
                if success:
                    updated_count += 1
                else:
                    skipped_count += 1
            else:
                print(f"⚠️  File not found: {filepath}")
                skipped_count += 1

    print(f"\n{'='*60}")
    print(f"✓ Updated: {updated_count} sessions")
    print(f"⚠ Skipped: {skipped_count} sessions")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
