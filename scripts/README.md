# Scripts

## expand_snippets.py

Expands `pymdownx.snippets` include syntax in Markdown files so they render correctly in VS Code and GitHub web (not just MkDocs).

### Usage

**Manual expansion:**
```bash
python scripts/expand_snippets.py protocol/process.md
```

**With pre-commit (automatic):**
```bash
# Install pre-commit (once)
pip install pre-commit

# Install the hooks (once)
pre-commit install

# Now it runs automatically on git commit
git add protocol/process.md
git commit -m "Update process doc"
```

The script will expand `--8<-- "md/pico_strategy.md:pico-table"` into the actual table content before committing, so the file works everywhere.

### How it works

1. Finds `--8<-- "path:section"` patterns
2. Reads the target file
3. Extracts content between `<!--8<-- [start:section] -->` and `<!--8<-- [end:section] -->` markers
4. Replaces the include line with the actual content
5. Adds `<!-- AUTO-EXPANDED -->` comments so you can see what was expanded
