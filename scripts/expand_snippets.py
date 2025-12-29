#!/usr/bin/env python3
"""
Expand pymdownx.snippets includes in Markdown files.
Converts --8<-- "file:section" syntax into the actual content.
"""
import re
import sys
from pathlib import Path


def extract_snippet(file_path: Path, section: str = None) -> str:
    """Extract content from a file, optionally between snippet markers."""
    if not file_path.exists():
        return f"<!-- ERROR: File not found: {file_path} -->"

    content = file_path.read_text(encoding='utf-8')

    if section:
        # Extract content between <!--8<-- [start:section] -->
        # and <!--8<-- [end:section] -->
        start_marker = f"<!--8<-- [start:{section}] -->"
        end_marker = f"<!--8<-- [end:{section}] -->"

        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)

        if start_idx == -1 or end_idx == -1:
            _msg = f"ERROR: Section '{section}' not found in {file_path}"
            return _msg

        # Extract content between markers
        snippet = content[start_idx + len(start_marker):end_idx].strip()
        return snippet
    else:
        # Return entire file content
        return content


def expand_snippets(md_file: Path, base_dir: Path) -> str:
    """Expand all snippet includes in a markdown file."""
    content = md_file.read_text(encoding='utf-8')

    # Pattern: --8<-- "path/to/file.md" or --8<-- "path/to/file.md:section"
    pattern = r'--8<--\s+"([^"]+)"'

    def replace_snippet(match):
        include_spec = match.group(1)

        # Parse path and optional section
        if ':' in include_spec:
            rel_path, section = include_spec.split(':', 1)
        else:
            rel_path, section = include_spec, None

        # Resolve relative to the markdown file's directory (or base_dir)
        target_file = (md_file.parent / rel_path).resolve()

        # Extract and return the snippet content
        snippet_content = extract_snippet(target_file, section)

        # Add a comment indicating this was auto-expanded
        header = f"<!-- AUTO-EXPANDED from {rel_path}{':' + section if section else ''} -->\n"
        footer = f"\n<!-- END AUTO-EXPANDED -->"

        return header + snippet_content + footer

    # Replace all snippet includes
    expanded = re.sub(pattern, replace_snippet, content)

    return expanded


def main():
    if len(sys.argv) < 2:
        print("Usage: python expand_snippets.py <markdown_file> [output_file]")
        print("If output_file is omitted, overwrites the input file.")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else input_file

    if not input_file.exists():
        print(f"Error: {input_file} not found")
        sys.exit(1)

    base_dir = input_file.parent
    expanded_content = expand_snippets(input_file, base_dir)

    output_file.write_text(expanded_content, encoding='utf-8')
    print(f"[OK] Expanded snippets in {input_file} -> {output_file}")


if __name__ == '__main__':
    main()
