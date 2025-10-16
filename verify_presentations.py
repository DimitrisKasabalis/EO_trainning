#!/usr/bin/env python3
"""
Verify presentation sources and check for Reveal.js format
"""

import os
from pathlib import Path

BASE = Path("/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil")

print("="*80)
print("PRESENTATION SOURCE VERIFICATION")
print("="*80)

# Check for QMD (Quarto Reveal.js) sources
print("\n1. CHECKING FOR QUARTO REVEAL.JS SOURCES (.qmd)\n")

qmd_files = []
for day_folder in ["DAY_2", "DAY_3", "DAY_4"]:
    day_path = BASE / day_folder
    if day_path.exists():
        for qmd in day_path.rglob("*.qmd"):
            # Read first 30 lines to check for revealjs
            try:
                with open(qmd, 'r') as f:
                    content = f.read(2000)
                    is_revealjs = 'revealjs' in content.lower()
                    qmd_files.append({
                        'path': str(qmd.relative_to(BASE)),
                        'day': day_folder,
                        'size': qmd.stat().st_size,
                        'revealjs': is_revealjs
                    })
            except:
                pass

if qmd_files:
    print(f"Found {len(qmd_files)} Quarto files:\n")
    for qmd in qmd_files:
        status = "✅ Reveal.js" if qmd['revealjs'] else "❓ Unknown format"
        print(f"  {status} - {qmd['path']}")
        print(f"          Size: {qmd['size']:,} bytes")
else:
    print("  ❌ No Quarto (.qmd) files found in DAY_2, DAY_3, DAY_4")

# Check for PDF presentations
print("\n2. CHECKING PDF PRESENTATIONS\n")

pdf_files = []
for day_folder in ["DAY_2", "DAY_3", "DAY_4"]:
    day_path = BASE / day_folder
    if day_path.exists():
        # Get PDFs in root
        for pdf in day_path.glob("*.pdf"):
            pdf_files.append({
                'path': str(pdf.relative_to(BASE)),
                'day': day_folder,
                'name': pdf.name,
                'size': pdf.stat().st_size
            })

if pdf_files:
    print(f"Found {len(pdf_files)} PDF files:\n")
    
    # Group by day
    for day in ["DAY_2", "DAY_3", "DAY_4"]:
        day_pdfs = [p for p in pdf_files if p['day'] == day]
        if day_pdfs:
            print(f"\n  {day} ({len(day_pdfs)} PDFs):")
            for pdf in day_pdfs:
                print(f"    📄 {pdf['name']}")
                print(f"       {pdf['size']:,} bytes ({pdf['size']/1024:.1f} KB)")

# Check presentation folders
print("\n3. CHECKING PRESENTATION FOLDER STRUCTURE\n")

for day_folder in ["DAY_2", "DAY_3", "DAY_4"]:
    day_path = BASE / day_folder
    print(f"\n  {day_folder}:")
    
    if not day_path.exists():
        print(f"    ❌ Folder does not exist")
        continue
    
    # Check for session folders
    session_dirs = sorted(day_path.glob("session*"))
    if session_dirs:
        print(f"    Found {len(session_dirs)} session folders:")
        for sess_dir in session_dirs:
            print(f"      📁 {sess_dir.name}/")
            
            # Check for presentation subfolders
            pres_folders = list(sess_dir.glob("presentation*"))
            if pres_folders:
                for pres_folder in pres_folders:
                    files = list(pres_folder.iterdir())
                    print(f"         📂 {pres_folder.name}/ ({len(files)} files)")
                    
                    # List files
                    for f in files:
                        if f.is_file():
                            icon = "📄" if f.suffix == ".qmd" else "📋"
                            print(f"            {icon} {f.name}")

# Check if PDFs match sessions
print("\n4. CONTENT ALIGNMENT CHECK\n")

# Define expected sessions per original agenda
expected_sessions = {
    "DAY_2": [
        "Session 1: Supervised Classification with Random Forest",
        "Session 2: Land Cover Classification Lab (Palawan)",
        "Session 3: Introduction to Deep Learning and CNNs",
        "Session 4: CNN Hands-on Lab"
    ],
    "DAY_3": [
        "Session 1: Semantic Segmentation with U-Net",
        "Session 2: Flood Mapping with U-Net and Sentinel-1",
        "Session 3: Object Detection Techniques",
        "Session 4: Feature/Object Detection from Sentinel Imagery"
    ],
    "DAY_4": [
        "Session 1: LSTMs for Time Series",
        "Session 2: LSTM Drought Monitoring Lab",
        "Session 3: Emerging AI/ML Trends (GeoFMs, SSL, XAI)",
        "Session 4: Synthesis, Q&A, and Pathway"
    ]
}

for day, sessions in expected_sessions.items():
    print(f"\n  {day}:")
    print(f"  Expected: {len(sessions)} sessions")
    
    # Check how many PDFs exist for this day
    day_pdfs = [p['name'].lower() for p in pdf_files if p['day'] == day]
    
    for i, session in enumerate(sessions, 1):
        session_lower = session.lower()
        keywords = session_lower.split()[1:3]  # Get key words
        
        # Check if any PDF matches
        matches = [pdf for pdf in day_pdfs if any(kw in pdf for kw in keywords if len(kw) > 3)]
        
        if matches:
            print(f"    ✅ Session {i}: Found PDF(s)")
        else:
            print(f"    ⚠️  Session {i}: No matching PDF found")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80 + "\n")

if qmd_files:
    print("✅ GOOD NEWS: Found Quarto source files!")
    print(f"   {len([q for q in qmd_files if q['revealjs']])} are Reveal.js presentations")
    print("\n📋 RECOMMENDED ACTION:")
    print("   1. Use the existing .qmd files (they're already Reveal.js)")
    print("   2. Copy them to course_site/dayX/presentations/")
    print("   3. Render them to HTML slides")
    print("   4. Keep PDFs as backup/download option")
else:
    print("⚠️  NO QUARTO SOURCE FILES FOUND")
    print("\n📋 RECOMMENDED ACTION:")
    print("   1. PDFs exist but may not be Reveal.js format")
    print("   2. Need to verify if PDFs are from PowerPoint or Reveal.js")
    print("   3. Consider converting to Quarto Reveal.js for consistency with Day 1")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80 + "\n")

if qmd_files:
    print("Since Quarto sources exist, we should:")
    print("")
    print("1. Copy .qmd files to course_site structure")
    print("2. Verify they render correctly")
    print("3. Integrate into session pages")
    print("4. Optionally keep PDFs as downloads")
else:
    print("Since only PDFs exist, we should:")
    print("")
    print("1. Copy PDFs to course_site structure")
    print("2. Link them from session pages as downloads")
    print("3. Consider creating Quarto versions later for consistency")

print("\n✅ Verification complete!")
