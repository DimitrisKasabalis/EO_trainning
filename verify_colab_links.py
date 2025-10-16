#!/usr/bin/env python3
"""
Verify all Google Colab links for notebook accessibility
"""

import os
import json
import re
from pathlib import Path

# Repository info
REPO_OWNER = "DimitrisKasabalis"
REPO_NAME = "EO_trainning"
BRANCH = "main"

# Base paths
COURSE_SITE_DIR = Path("course_site")

def find_all_notebooks():
    """Find all .ipynb files in the project"""
    notebooks = []
    for root, dirs, files in os.walk(COURSE_SITE_DIR):
        # Skip .quarto directories
        if '.quarto' in root:
            continue
        for file in files:
            if file.endswith('.ipynb'):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(COURSE_SITE_DIR)
                notebooks.append({
                    'filename': file,
                    'full_path': str(full_path),
                    'rel_path': str(rel_path),
                    'exists': full_path.exists(),
                    'size': full_path.stat().st_size if full_path.exists() else 0
                })
    return notebooks

def find_colab_links():
    """Find all Colab links in QMD and MD files"""
    colab_links = []
    pattern = re.compile(r'https://colab\.research\.google\.com/github/([^/]+)/([^/]+)/blob/([^/]+)/(.+?\.ipynb)')

    for root, dirs, files in os.walk(COURSE_SITE_DIR):
        if '.quarto' in root:
            continue
        for file in files:
            if file.endswith(('.qmd', '.md')):
                file_path = Path(root) / file
                try:
                    content = file_path.read_text(encoding='utf-8')
                    matches = pattern.finditer(content)
                    for match in matches:
                        owner, repo, branch, notebook_path = match.groups()
                        colab_links.append({
                            'source_file': str(file_path.relative_to(COURSE_SITE_DIR)),
                            'colab_url': match.group(0),
                            'owner': owner,
                            'repo': repo,
                            'branch': branch,
                            'notebook_path': notebook_path
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return colab_links

def check_notebook_metadata(notebook_path):
    """Check if notebook has proper metadata for Colab"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        metadata = nb.get('metadata', {})

        return {
            'has_colab_metadata': 'colab' in metadata,
            'kernel': metadata.get('kernelspec', {}).get('name', 'unknown'),
            'language': metadata.get('language_info', {}).get('name', 'unknown'),
            'cells': len(nb.get('cells', [])),
            'colab_settings': metadata.get('colab', {})
        }
    except Exception as e:
        return {
            'error': str(e)
        }

def generate_expected_colab_url(notebook_rel_path):
    """Generate expected Colab URL for a notebook"""
    # Try different possible paths in GitHub
    possible_paths = [
        notebook_rel_path,  # course_site/day1/notebooks/...
        notebook_rel_path.replace('course_site/', ''),  # day1/notebooks/...
        f"notebooks/{Path(notebook_rel_path).name}",  # notebooks/...
    ]

    urls = []
    for path in possible_paths:
        url = f"https://colab.research.google.com/github/{REPO_OWNER}/{REPO_NAME}/blob/{BRANCH}/{path}"
        urls.append(url)

    return urls

def main():
    print("=" * 80)
    print("GOOGLE COLAB NOTEBOOK VERIFICATION")
    print("=" * 80)
    print()

    # Find all notebooks
    print("📓 Finding all notebooks...")
    notebooks = find_all_notebooks()
    print(f"   Found {len(notebooks)} notebooks\n")

    # Find all Colab links
    print("🔗 Finding all Colab links in QMD/MD files...")
    colab_links = find_colab_links()
    print(f"   Found {len(colab_links)} Colab links\n")

    # Organize by day
    print("=" * 80)
    print("NOTEBOOKS BY DAY")
    print("=" * 80)
    print()

    for day in ['day1', 'day2', 'day3', 'day4']:
        day_notebooks = [nb for nb in notebooks if day in nb['rel_path']]
        if day_notebooks:
            print(f"\n📘 {day.upper()}: {len(day_notebooks)} notebooks")
            print("-" * 80)
            for nb in day_notebooks:
                print(f"   ✓ {nb['filename']}")
                print(f"     Path: {nb['rel_path']}")
                print(f"     Size: {nb['size']:,} bytes")

                # Check metadata
                metadata = check_notebook_metadata(nb['full_path'])
                if 'error' not in metadata:
                    print(f"     Cells: {metadata['cells']}")
                    print(f"     Kernel: {metadata['kernel']}")
                    print(f"     Colab metadata: {'✓' if metadata['has_colab_metadata'] else '✗'}")

                # Generate expected URLs
                expected_urls = generate_expected_colab_url(nb['rel_path'])
                print(f"     Expected Colab URLs:")
                for url in expected_urls[:2]:  # Show first 2 options
                    print(f"       - {url}")
                print()

    # Check Colab links
    print("=" * 80)
    print("COLAB LINKS FOUND IN DOCUMENTATION")
    print("=" * 80)
    print()

    for link in colab_links:
        print(f"📄 Source: {link['source_file']}")
        print(f"   Notebook: {link['notebook_path']}")
        print(f"   URL: {link['colab_url']}")

        # Check if path matches any actual notebook
        notebook_name = Path(link['notebook_path']).name
        matching_notebooks = [nb for nb in notebooks if nb['filename'] == notebook_name]

        if matching_notebooks:
            print(f"   ✓ Notebook exists: {matching_notebooks[0]['rel_path']}")
        else:
            print(f"   ✗ WARNING: Notebook not found!")
        print()

    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    print(f"Total notebooks: {len(notebooks)}")
    print(f"Total Colab links: {len(colab_links)}")

    # Check for notebooks without Colab links
    linked_notebook_names = {Path(link['notebook_path']).name for link in colab_links}
    unlinked_notebooks = [nb for nb in notebooks if nb['filename'] not in linked_notebook_names]

    if unlinked_notebooks:
        print(f"\n⚠️  Notebooks WITHOUT Colab links: {len(unlinked_notebooks)}")
        for nb in unlinked_notebooks:
            print(f"   - {nb['rel_path']}")
    else:
        print("\n✓ All notebooks have Colab links!")

    # Check for broken links
    broken_links = []
    for link in colab_links:
        notebook_name = Path(link['notebook_path']).name
        matching_notebooks = [nb for nb in notebooks if nb['filename'] == notebook_name]
        if not matching_notebooks:
            broken_links.append(link)

    if broken_links:
        print(f"\n❌ Broken Colab links: {len(broken_links)}")
        for link in broken_links:
            print(f"   - {link['notebook_path']} (referenced in {link['source_file']})")
    else:
        print("\n✓ No broken Colab links detected!")

    print("\n" + "=" * 80)
    print("RECOMMENDED ACTIONS")
    print("=" * 80)
    print()
    print("To verify notebooks open in Colab:")
    print()
    print("1. Ensure all notebooks are committed and pushed to GitHub")
    print("2. Test each Colab URL in a browser")
    print("3. Check that notebooks are in the correct GitHub path")
    print()
    print("Common GitHub paths to try:")
    print(f"   - https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}/course_site/dayX/notebooks/")
    print(f"   - https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}/dayX/notebooks/")
    print(f"   - https://github.com/{REPO_OWNER}/{REPO_NAME}/tree/{BRANCH}/notebooks/")
    print()

if __name__ == "__main__":
    main()
