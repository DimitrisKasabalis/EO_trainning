#!/bin/bash
# Integrate presentations into course_site
# Execute Phase 1: Copy presentations and create folder structure

echo "=========================================="
echo "PRESENTATION INTEGRATION - PHASE 1"
echo "=========================================="
echo ""

# Step 1: Create presentation folders
echo "Step 1: Creating presentation folders..."
mkdir -p course_site/day2/presentations
mkdir -p course_site/day3/presentations  
mkdir -p course_site/day4/presentations
echo "✅ Folders created"
echo ""

# Step 2: Copy Day 2 Reveal.js source
echo "Step 2: Copying Day 2 Session 1 Reveal.js..."
cp DAY_2/session1/presentation/session1_random_forest.qmd \
   course_site/day2/presentations/
cp DAY_2/session1/presentation/custom.scss \
   course_site/day2/presentations/
echo "✅ Reveal.js source copied"
echo ""

# Step 3: Copy all PDFs
echo "Step 3: Copying PDF presentations..."

echo "  Day 2 PDFs..."
cp DAY_2/*.pdf course_site/day2/presentations/ 2>/dev/null
echo "  ✅ Day 2: $(ls course_site/day2/presentations/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs"

echo "  Day 3 PDFs..."
cp DAY_3/*.pdf course_site/day3/presentations/ 2>/dev/null
echo "  ✅ Day 3: $(ls course_site/day3/presentations/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs"

echo "  Day 4 PDFs..."
cp DAY_4/*.pdf course_site/day4/presentations/ 2>/dev/null
echo "  ✅ Day 4: $(ls course_site/day4/presentations/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDFs"
echo ""

# Step 4: Verify files copied
echo "Step 4: Verification..."
echo "  Day 2 presentations folder:"
ls -lh course_site/day2/presentations/ | tail -n +2 | wc -l | xargs echo "    Files:"
echo "  Day 3 presentations folder:"
ls -lh course_site/day3/presentations/ | tail -n +2 | wc -l | xargs echo "    Files:"
echo "  Day 4 presentations folder:"
ls -lh course_site/day4/presentations/ | tail -n +2 | wc -l | xargs echo "    Files:"
echo ""

echo "=========================================="
echo "✅ INTEGRATION COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Render Day 2 Session 1 Reveal.js: cd course_site/day2/presentations && quarto render session1_random_forest.qmd"
echo "2. Update session QMD files to link presentations"
echo "3. Update day index pages"
echo ""
