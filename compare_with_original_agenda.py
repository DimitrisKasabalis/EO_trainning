#!/usr/bin/env python3
"""
Compare implemented course structure with original training agenda
"""

import re
from pathlib import Path

# Read the original agenda
with open('/tmp/original_agenda.txt', 'r') as f:
    agenda_text = f.read()

print("="*80)
print("COPPHIL COURSE IMPLEMENTATION vs ORIGINAL AGENDA COMPARISON")
print("="*80)

# Parse original agenda structure
print("\n" + "="*80)
print("ORIGINAL AGENDA STRUCTURE")
print("="*80 + "\n")

# Extract Day sections
day_pattern = r'Day\s+(\d+)[:\s]+(.*?)(?=Day\s+\d+:|$)'
days = re.findall(day_pattern, agenda_text, re.DOTALL | re.IGNORECASE)

original_structure = {}

for day_num, day_content in days:
    print(f"\n{'='*80}")
    print(f"DAY {day_num}")
    print(f"{'='*80}\n")
    
    # Extract title
    title_match = re.search(r'^(.*?)(?:Session|•)', day_content.strip(), re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
        print(f"Title: {title}\n")
    
    # Extract sessions
    session_pattern = r'Session\s+(\d+):\s+(.*?)(?=Session\s+\d+:|Day\s+\d+:|$)'
    sessions = re.findall(session_pattern, day_content, re.DOTALL | re.IGNORECASE)
    
    original_structure[f"day{day_num}"] = {
        "title": title if title_match else "",
        "sessions": {}
    }
    
    print(f"Sessions Found: {len(sessions)}\n")
    
    for sess_num, sess_content in sessions:
        # Get session title (first line)
        sess_lines = sess_content.strip().split('\n')
        sess_title = sess_lines[0].strip() if sess_lines else ""
        
        # Remove leading/trailing special chars
        sess_title = re.sub(r'^[•\s]+', '', sess_title)
        sess_title = sess_title.split('(')[0].strip()  # Remove duration info
        
        print(f"  Session {sess_num}: {sess_title}")
        
        original_structure[f"day{day_num}"]["sessions"][f"session{sess_num}"] = {
            "title": sess_title,
            "content": sess_content
        }
        
        # Check for key elements
        has_platform = 'Platform:' in sess_content or 'Colab' in sess_content
        has_data = 'Data:' in sess_content or 'dataset' in sess_content.lower()
        has_case_study = 'Case Study:' in sess_content
        has_workflow = 'Workflow:' in sess_content
        
        elements = []
        if has_platform: elements.append("Platform")
        if has_data: elements.append("Data")
        if has_case_study: elements.append("Case Study")
        if has_workflow: elements.append("Workflow")
        
        if elements:
            print(f"     Elements: {', '.join(elements)}")
    
    print()

print("\n" + "="*80)
print("IMPLEMENTED COURSE STRUCTURE")
print("="*80 + "\n")

# Now check what we have implemented
course_site = Path("/Users/dimitriskasampalis/Projects/Neuralio/ESAPhil/course_site")

implemented_structure = {}

for day_dir in sorted(course_site.glob("day*")):
    day_name = day_dir.name
    day_num = day_name.replace("day", "")
    
    print(f"\n{'='*80}")
    print(f"DAY {day_num}")
    print(f"{'='*80}\n")
    
    # Read index to get title
    index_file = day_dir / "index.qmd"
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_content = f.read()
            # Extract title
            title_match = re.search(r'title:\s*"([^"]+)"', index_content)
            if title_match:
                print(f"Title: {title_match.group(1)}\n")
    
    # Check sessions
    sessions_dir = day_dir / "sessions"
    if sessions_dir.exists():
        session_files = sorted(sessions_dir.glob("session*.qmd"))
        print(f"Sessions Implemented: {len(session_files)}\n")
        
        for sess_file in session_files:
            sess_num = sess_file.stem.replace("session", "")
            
            # Read session file
            with open(sess_file, 'r') as f:
                sess_content = f.read()
                
                # Extract title
                title_match = re.search(r'title:\s*"([^"]+)"', sess_content)
                if title_match:
                    sess_title = title_match.group(1)
                    print(f"  Session {sess_num}: {sess_title}")
                    
                    # Check for key elements
                    has_platform = 'Colab' in sess_content or 'Platform' in sess_content
                    has_case_study = 'Case Study' in sess_content or 'Philippine' in sess_content
                    has_workflow = 'Workflow' in sess_content or 'workflow' in sess_content.lower()
                    has_notebook = (day_dir / "notebooks").exists()
                    
                    elements = []
                    if has_platform: elements.append("Platform")
                    if has_case_study: elements.append("Case Study")
                    if has_workflow: elements.append("Workflow")
                    if has_notebook: elements.append("Notebook")
                    
                    if elements:
                        print(f"     Elements: {', '.join(elements)}")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80 + "\n")

# Compare structure
print("DAY-BY-DAY COMPARISON:\n")

for day in ["day1", "day2", "day3", "day4"]:
    day_num = day.replace("day", "")
    print(f"\nDay {day_num}:")
    
    if day in original_structure:
        orig_sessions = len(original_structure[day]["sessions"])
        print(f"  Original Agenda: {orig_sessions} sessions")
        
        # Check implemented
        day_dir = course_site / day / "sessions"
        if day_dir.exists():
            impl_sessions = len(list(day_dir.glob("session*.qmd")))
            print(f"  Implemented: {impl_sessions} sessions")
            
            if impl_sessions == orig_sessions:
                print(f"  Status: ✅ MATCH - All sessions implemented")
            elif impl_sessions > orig_sessions:
                print(f"  Status: ⚠️  MORE sessions implemented than planned")
            else:
                print(f"  Status: ❌ MISSING {orig_sessions - impl_sessions} session(s)")
        else:
            print(f"  Implemented: 0 sessions")
            print(f"  Status: ❌ NOT IMPLEMENTED")

print("\n" + "="*80)
print("KEY FEATURES COMPARISON")
print("="*80 + "\n")

key_features = [
    ("Palawan Land Cover", "Day 2"),
    ("Central Luzon Flood", "Day 3"),
    ("Metro Manila Urban", "Day 3"),
    ("Mindanao Drought", "Day 4"),
    ("Google Colab", "All days"),
    ("PhilSA/DOST-ASTI", "Day 1"),
    ("Random Forest", "Day 2"),
    ("U-Net", "Day 3"),
    ("LSTM", "Day 4"),
    ("Foundation Models", "Day 4")
]

print("Feature Coverage Check:\n")

for feature, expected_day in key_features:
    # Check if feature appears in agenda
    in_agenda = feature.lower() in agenda_text.lower()
    
    # Check if implemented
    found_in_course = False
    for day_dir in course_site.glob("day*"):
        for qmd_file in day_dir.rglob("*.qmd"):
            with open(qmd_file, 'r') as f:
                if feature.lower() in f.read().lower():
                    found_in_course = True
                    break
        if found_in_course:
            break
    
    status = "✅" if (in_agenda and found_in_course) else ("⚠️" if in_agenda else "❌")
    impl_status = "✅ Implemented" if found_in_course else "❌ Not found"
    
    print(f"  {status} {feature:25s} ({expected_day:10s}) - {impl_status}")

print("\n" + "="*80)
print("FINAL ASSESSMENT")
print("="*80 + "\n")

total_orig_sessions = sum(len(day_data["sessions"]) for day_data in original_structure.values())
total_impl_sessions = sum(len(list((course_site / day / "sessions").glob("session*.qmd"))) 
                         for day in ["day1", "day2", "day3", "day4"] 
                         if (course_site / day / "sessions").exists())

print(f"Total Sessions in Original Agenda: {total_orig_sessions}")
print(f"Total Sessions Implemented: {total_impl_sessions}")
print(f"Match: {'✅ YES' if total_impl_sessions == total_orig_sessions else '⚠️  NO'}")

print(f"\nOverall Alignment: ", end="")
if total_impl_sessions == total_orig_sessions:
    print("✅ FULLY ALIGNED - Course structure matches original agenda")
elif total_impl_sessions > total_orig_sessions:
    print("⚠️  ENHANCED - More sessions than originally planned")
else:
    print(f"❌ INCOMPLETE - Missing {total_orig_sessions - total_impl_sessions} session(s)")

print("\n" + "="*80)
