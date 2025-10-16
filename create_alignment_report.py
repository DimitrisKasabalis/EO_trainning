#!/usr/bin/env python3
"""
Comprehensive Course Alignment Verification
Checks all QMD files, notebooks, and documentation are properly aligned
"""

import os
import json
from pathlib import Path

# Base directory
BASE_DIR = Path("/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site")

# Expected structure
EXPECTED_STRUCTURE = {
    "day1": {
        "sessions": ["session1.qmd", "session2.qmd", "session3.qmd", "session4.qmd"],
        "notebooks": ["Day1_Session3_Python_Geospatial_Data.ipynb", "Day1_Session4_Google_Earth_Engine.ipynb"],
        "index": "index.qmd"
    },
    "day2": {
        "sessions": ["session1.qmd", "session2.qmd", "session3.qmd", "session4.qmd"],
        "notebooks": [
            "session1_hands_on_lab_student.ipynb",
            "session1_theory_notebook_STUDENT.ipynb",
            "session2_extended_lab_STUDENT.ipynb",
            "session3_theory_interactive.ipynb",
            "session4_cnn_classification_STUDENT.ipynb",
            "session4_transfer_learning_STUDENT.ipynb",
            "session4_unet_segmentation_STUDENT.ipynb"
        ],
        "index": "index.qmd"
    },
    "day3": {
        "sessions": ["session1.qmd", "session2.qmd", "session3.qmd", "session4.qmd"],
        "notebooks": [
            "Day3_Session2_Flood_Mapping_UNet.ipynb",
            "Day3_Session4_Object_Detection_STUDENT.ipynb"
        ],
        "index": "index.qmd",
        "guides": ["DATA_GUIDE.md"]
    },
    "day4": {
        "sessions": ["session1.qmd", "session2.qmd", "session3.qmd", "session4.qmd"],
        "notebooks": [
            "day4_session1_lstm_demo_STUDENT.ipynb",
            "day4_session2_lstm_drought_lab_STUDENT.ipynb"
        ],
        "index": "index.qmd"
    }
}

def check_file_exists(filepath):
    """Check if file exists and return size"""
    if filepath.exists():
        size = filepath.stat().st_size
        return True, size
    return False, 0

def check_notebook_cells(notebook_path):
    """Count cells in Jupyter notebook"""
    try:
        with open(notebook_path, 'r') as f:
            nb = json.load(f)
            return len(nb.get('cells', []))
    except:
        return 0

def verify_day(day_name, structure):
    """Verify a single day's structure"""
    print(f"\n{'='*80}")
    print(f"📋 DAY {day_name.upper().replace('DAY', '')} VERIFICATION")
    print(f"{'='*80}\n")
    
    day_dir = BASE_DIR / day_name
    results = {
        "day": day_name,
        "sessions": {"expected": 0, "found": 0, "missing": [], "sizes": {}},
        "notebooks": {"expected": 0, "found": 0, "missing": [], "cells": {}},
        "index": {"exists": False, "size": 0},
        "guides": {"expected": 0, "found": 0, "missing": []}
    }
    
    # Check index
    index_path = day_dir / structure["index"]
    exists, size = check_file_exists(index_path)
    results["index"]["exists"] = exists
    results["index"]["size"] = size
    
    status = "✅" if exists else "❌"
    print(f"{status} Index Page: {structure['index']}")
    if exists:
        print(f"   Size: {size:,} bytes ({size/1024:.1f} KB)")
    else:
        print(f"   ⚠️  MISSING!")
    
    # Check sessions
    print(f"\n📄 Session QMD Files:")
    sessions_dir = day_dir / "sessions"
    results["sessions"]["expected"] = len(structure["sessions"])
    
    for session_file in structure["sessions"]:
        session_path = sessions_dir / session_file
        exists, size = check_file_exists(session_path)
        
        if exists:
            results["sessions"]["found"] += 1
            results["sessions"]["sizes"][session_file] = size
            print(f"   ✅ {session_file:30s} - {size:,} bytes ({size/1024:.1f} KB)")
        else:
            results["sessions"]["missing"].append(session_file)
            print(f"   ❌ {session_file:30s} - MISSING")
    
    # Check notebooks
    print(f"\n📓 Notebooks:")
    notebooks_dir = day_dir / "notebooks"
    results["notebooks"]["expected"] = len(structure["notebooks"])
    
    for notebook_file in structure["notebooks"]:
        notebook_path = notebooks_dir / notebook_file
        exists, size = check_file_exists(notebook_path)
        
        if exists:
            results["notebooks"]["found"] += 1
            cells = check_notebook_cells(notebook_path)
            results["notebooks"]["cells"][notebook_file] = cells
            print(f"   ✅ {notebook_file:50s}")
            print(f"      Size: {size:,} bytes ({size/1024:.1f} KB), Cells: {cells}")
        else:
            results["notebooks"]["missing"].append(notebook_file)
            print(f"   ❌ {notebook_file:50s} - MISSING")
    
    # Check guides (Day 3)
    if "guides" in structure:
        print(f"\n📖 Guides & Documentation:")
        results["guides"]["expected"] = len(structure["guides"])
        
        for guide_file in structure["guides"]:
            guide_path = day_dir / guide_file
            exists, size = check_file_exists(guide_path)
            
            if exists:
                results["guides"]["found"] += 1
                # Count lines
                with open(guide_path, 'r') as f:
                    lines = len(f.readlines())
                print(f"   ✅ {guide_file:30s} - {size:,} bytes ({size/1024:.1f} KB), {lines} lines")
            else:
                results["guides"]["missing"].append(guide_file)
                print(f"   ❌ {guide_file:30s} - MISSING")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Index Page: {'✅ Present' if results['index']['exists'] else '❌ Missing'}")
    print(f"   Sessions: {results['sessions']['found']}/{results['sessions']['expected']} " +
          f"({'✅' if results['sessions']['found'] == results['sessions']['expected'] else '⚠️'})")
    print(f"   Notebooks: {results['notebooks']['found']}/{results['notebooks']['expected']} " +
          f"({'✅' if results['notebooks']['found'] == results['notebooks']['expected'] else '⚠️'})")
    
    if "guides" in structure:
        print(f"   Guides: {results['guides']['found']}/{results['guides']['expected']} " +
              f"({'✅' if results['guides']['found'] == results['guides']['expected'] else '⚠️'})")
    
    return results

def main():
    print("\n" + "="*80)
    print("🔍 COPPHIL COURSE ALIGNMENT VERIFICATION")
    print("="*80)
    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Verification Time: {Path(__file__).stat().st_mtime}")
    
    all_results = {}
    
    # Check each day
    for day_name, structure in EXPECTED_STRUCTURE.items():
        results = verify_day(day_name, structure)
        all_results[day_name] = results
    
    # Overall Summary
    print(f"\n{'='*80}")
    print(f"🎯 OVERALL COURSE STATUS")
    print(f"{'='*80}\n")
    
    total_sessions_expected = sum(r["sessions"]["expected"] for r in all_results.values())
    total_sessions_found = sum(r["sessions"]["found"] for r in all_results.values())
    total_notebooks_expected = sum(r["notebooks"]["expected"] for r in all_results.values())
    total_notebooks_found = sum(r["notebooks"]["found"] for r in all_results.values())
    
    print(f"📄 Session QMD Files: {total_sessions_found}/{total_sessions_expected} " +
          f"({100*total_sessions_found/total_sessions_expected:.0f}%)")
    print(f"📓 Notebooks: {total_notebooks_found}/{total_notebooks_expected} " +
          f"({100*total_notebooks_found/total_notebooks_expected:.0f}%)")
    
    # Day-by-day status
    print(f"\n📅 Day-by-Day Completion:")
    for day_name, results in all_results.items():
        day_num = day_name.replace('day', '').upper()
        sessions_complete = results["sessions"]["found"] == results["sessions"]["expected"]
        notebooks_complete = results["notebooks"]["found"] == results["notebooks"]["expected"]
        index_complete = results["index"]["exists"]
        
        if sessions_complete and notebooks_complete and index_complete:
            status = "✅ 100%"
        elif sessions_complete and index_complete:
            status = "⚠️  90%"
        else:
            status = "⚠️  <90%"
        
        print(f"   Day {day_num}: {status} - Sessions: {results['sessions']['found']}/{results['sessions']['expected']}, " +
              f"Notebooks: {results['notebooks']['found']}/{results['notebooks']['expected']}")
    
    # Issues found
    print(f"\n⚠️  Issues Found:")
    issues_found = False
    
    for day_name, results in all_results.items():
        if results["sessions"]["missing"]:
            issues_found = True
            print(f"\n   {day_name.upper()} - Missing Sessions:")
            for missing in results["sessions"]["missing"]:
                print(f"      ❌ {missing}")
        
        if results["notebooks"]["missing"]:
            issues_found = True
            print(f"\n   {day_name.upper()} - Missing Notebooks:")
            for missing in results["notebooks"]["missing"]:
                print(f"      ❌ {missing}")
        
        if not results["index"]["exists"]:
            issues_found = True
            print(f"\n   {day_name.upper()} - Missing Index")
    
    if not issues_found:
        print(f"   ✅ No issues found - All files present!")
    
    # Detailed notebook info
    print(f"\n{'='*80}")
    print(f"📊 NOTEBOOK DETAILS")
    print(f"{'='*80}\n")
    
    for day_name, results in all_results.items():
        if results["notebooks"]["cells"]:
            print(f"{day_name.upper()}:")
            for nb_file, cells in results["notebooks"]["cells"].items():
                print(f"   {nb_file:50s} - {cells:3d} cells")
    
    print(f"\n{'='*80}")
    print(f"✅ VERIFICATION COMPLETE")
    print(f"{'='*80}\n")
    
    # Return summary
    return {
        "total_sessions": f"{total_sessions_found}/{total_sessions_expected}",
        "total_notebooks": f"{total_notebooks_found}/{total_notebooks_expected}",
        "completion": f"{100*(total_sessions_found + total_notebooks_found)/(total_sessions_expected + total_notebooks_expected):.0f}%"
    }

if __name__ == "__main__":
    summary = main()
    print(f"Course Completion: {summary['completion']}")
