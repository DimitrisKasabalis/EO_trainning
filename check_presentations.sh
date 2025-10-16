#!/bin/bash
echo "PRESENTATION CHECK"
echo "=================="
echo ""
echo "Day 1 (course_site/day1/presentations):"
ls course_site/day1/presentations/*.qmd 2>/dev/null | wc -l
echo ""
echo "Day 2 (course_site/day2/presentations):"
ls course_site/day2/presentations/*.qmd 2>/dev/null || echo "No folder"
echo ""
echo "Day 3 (course_site/day3/presentations):"
ls course_site/day3/presentations/*.qmd 2>/dev/null || echo "No folder"
echo ""
echo "Day 4 (course_site/day4/presentations):"
ls course_site/day4/presentations/*.qmd 2>/dev/null || echo "No folder"
echo ""
echo "DAY_3 folder presentations:"
find DAY_3 -name "*.pdf" -path "*/presentation*" 2>/dev/null | wc -l
