# Repository Guidelines

## Project Structure & Module Organization
- `course_site/` is the canonical Quarto project; sessions, notebooks, and resources inside feed the multi-day website.
- `DAY_1/` captures day-one production assets: keep raw docs in `source_materials/` and deliverables in `training_output/` (`notebooks/`, `presentations/`, `datasets/`, `handouts/`).
- `Course_Materials/` stores reusable walkthrough examples, while `Docs/` tracks high-resolution references.
- Plan for additional days by cloning the `DAY_1` layout into `DAY_2/`, `DAY_3/`, etc., leaving the shared website in `course_site/`.

## Build, Test, and Development Commands
- `quarto check course_site` verifies tooling, extensions, and YAML before publishing.
- `quarto render course_site` rebuilds the `_site/` output; append `--watch` during active editing.
- `quarto preview course_site` launches a live preview (remember to stop it so temporary files are cleaned up).
- From `DAY_1/training_output/notebooks`, run `jupyter nbconvert --execute Day1_Session3_Python_Geospatial_Data.ipynb --to notebook` to confirm execution paths; repeat for other notebooks as needed.

## Coding Style & Naming Conventions
- Keep `.qmd` filenames lowercase with underscores; start major sections at `##` to keep navigation clean.
- Follow PEP 8 within notebooks (4-space indentation, `snake_case` identifiers); explain non-obvious EO operations with concise inline comments.
- Store shared styling in `course_site/styles` and global media in `course_site/images`; reference via relative paths from content files.

## Testing Guidelines
- Execute the nbconvert checks after code or data updates, and re-run notebooks that reach out to external APIs to surface credential issues early.
- Spot-check rendered `_site/` pages for broken links, download buttons, and syntax highlighting before merges.
- Regenerate `handouts/` PDFs whenever their source Markdown changes and confirm timestamps align with published artefacts.

## Commit & Pull Request Guidelines
- Write imperative commit subjects with scope prefixes (e.g., `docs: refresh session2 narrative`, `notebooks: add gee export example`); separate binary artefacts from text updates when feasible.
- In PR descriptions, summarise intent, affected directories, verification commands, and any follow-on work; link external tracker IDs where applicable.
- Attach screenshots or URLs for visual updates and call out large files that reviewers must regenerate locally.

## Data & Asset Handling
- Keep proprietary datasets untracked; document acquisition steps in `course_site/resources` or the relevant handout instead.
- Retain Word originals in `source_materials/original_docs/` and re-export Markdown when source content changes to preserve traceability.
