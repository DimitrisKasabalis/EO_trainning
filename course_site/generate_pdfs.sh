#!/bin/bash
# Generate PDFs for all session files

echo "Generating PDFs for all sessions..."
echo "===================================="

# Day 1
echo "Day 1..."
cd day1/sessions
quarto render session1.qmd --to pdf
quarto render session2.qmd --to pdf
quarto render session3.qmd --to pdf
quarto render session4.qmd --to pdf
mv session1.pdf session2.pdf session3.pdf session4.pdf ../pdfs/
cd ../..

# Day 2
echo "Day 2..."
cd day2/sessions
quarto render session1.qmd --to pdf
quarto render session2.qmd --to pdf
quarto render session3.qmd --to pdf
quarto render session4.qmd --to pdf
mv session1.pdf session2.pdf session3.pdf session4.pdf ../pdfs/
cd ../..

# Day 3
echo "Day 3..."
cd day3/sessions
quarto render session1.qmd --to pdf
quarto render session2.qmd --to pdf
quarto render session3.qmd --to pdf
quarto render session4.qmd --to pdf
mv session1.pdf session2.pdf session3.pdf session4.pdf ../pdfs/
cd ../..

# Day 4
echo "Day 4..."
cd day4/sessions
quarto render session1.qmd --to pdf
quarto render session2.qmd --to pdf
quarto render session3.qmd --to pdf
quarto render session4.qmd --to pdf
mv session1.pdf session2.pdf session3.pdf session4.pdf ../pdfs/
cd ../..

echo "===================================="
echo "✓ All PDFs generated successfully!"
echo ""
echo "PDF locations:"
ls -lh day*/pdfs/*.pdf
