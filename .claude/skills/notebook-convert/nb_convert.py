# !/usr/bin/env python3
"""
Notebook Conversion Skill for Maritime Graph Toolkit

Converts Jupyter notebooks from docs/notebooks to docs/notebooks/dev for development/testing.
Supports conversion to .ipynb (copy), .py (Python script), or .md (Markdown) formats.
Provides flags for conversion, cleanup, checking, and listing notebooks.

Usage:
    python nb_convert.py --all                           # Convert all notebooks (as .ipynb)
    python nb_convert.py --all --to-python               # Convert all to .py
    python nb_convert.py --all --to-markdown             # Convert all to .md
    python nb_convert.py --notebook-name "graph"         # Convert specific notebook(s) by pattern
    python nb_convert.py --notebook-name "graph" --to-python  # Convert to .py by pattern
    python nb_convert.py --cleanup                       # Remove dev folder and all converted files
    python nb_convert.py --check                         # Show what would be converted (dry run)
    python nb_convert.py --list                          # List all available notebooks
"""

import argparse
import shutil
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path
import tempfile
import difflib
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


class FileFormat(Enum):
    """Supported file formats for sync."""
    IPYNB = '.ipynb'
    PYTHON = '.py'
    MARKDOWN = '.md'


@dataclass
class SyncPair:
    """Represents a pair of files to sync."""
    notebook_stem: str
    source_ipynb: Path
    dev_file: Path
    format: FileFormat

    @property
    def source_exists(self) -> bool:
        return self.source_ipynb.exists()

    @property
    def dev_exists(self) -> bool:
        return self.dev_file.exists()

class MergeDirection(Enum):
    """Merge direction for sync operation."""
    DEV_TO_SOURCE = "dev-to-source"
    SOURCE_TO_DEV = "source-to-dev"
    AUTO = "auto"  # Choose based on timestamps


class NotebookConverter:
    """Manages notebook conversion between source and dev directories."""

    # Class constants
    CONVERSION_TIMEOUT = 30  # seconds
    MAX_DIFF_PREVIEW_LINES = 50

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the converter with project paths.

        Args:
            project_root: Project root directory. If None, auto-detects from script location.
        """
        if project_root is None:
            # Auto-detect: script is in .claude/skills/notebook-convert/
            # Project root is 3 levels up
            self.project_root = Path(__file__).resolve().parents[3]
        else:
            self.project_root = Path(project_root)

        self.source_dir = self.project_root / "docs" / "notebooks"
        self.dev_dir = self.source_dir / "dev"
        self.changelog_path = self.dev_dir / "NB_CHANGELOG.md"
        self.readme_path = self.dev_dir / "README.md"

        # Check nbconvert availability
        self.nbconvert_available = self._check_nbconvert_available()

    def ensure_dev_dir(self) -> None:
        """Create dev directory if it doesn't exist, with README and CHANGELOG."""
        if not self.dev_dir.exists():
            self.dev_dir.mkdir(parents=True, exist_ok=True)
            self._create_readme()
            self._create_changelog()
            print(f"✓ Created dev directory: {self.dev_dir}")

    def _create_readme(self) -> None:
        """Create README.md in dev directory."""
        readme_content = """# Development Notebooks

This directory contains working copies of notebooks from `docs/notebooks/` for testing and development.

## ⚠️ Important Notes

- **Do NOT commit notebooks from this directory** - they are for local development only
- Changes should be tested here, then applied to the original notebooks in parent directory
- This directory and its contents are ignored by git (see `.gitignore`)
- Files are automatically cleaned up with `--cleanup` flag

## Purpose

Use this directory to:
- Test notebook modifications without affecting originals
- Experiment with cell execution and outputs
- Validate workflow changes before committing

## Workflow

1. Convert notebooks with `python .claude/skills/notebook-convert/nb_convert.py --notebook-name <pattern>`
2. Open and edit notebooks in this directory
3. Test changes thoroughly
4. Apply validated changes to original notebooks in parent directory
5. Clean up with `--cleanup` flag when done

## Tracking Changes

See `NB_CHANGELOG.md` for conversion history.
"""
        self.readme_path.write_text(readme_content)
        print(f"✓ Created README: {self.readme_path}")

    def _create_changelog(self) -> None:
        """Create NB_CHANGELOG.md in dev directory."""
        changelog_content = f"""# Notebook Conversion Changelog

This file tracks conversion operations for development notebooks.

---

## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Initial Setup

- Created dev directory structure
- Added README.md
- Initialized NB_CHANGELOG.md
"""
        self.changelog_path.write_text(changelog_content)
        print(f"✓ Created NB_CHANGELOG: {self.changelog_path}")

    def _append_changelog(self, operation: str, files: List[str]) -> None:
        """
        Append operation to changelog.

        Args:
            operation: Description of the operation (e.g., "Converted", "Cleaned up")
            files: List of affected filenames
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"\n## {timestamp} - {operation}\n\n"
        entry += "\n".join(f"- {fname}" for fname in files)
        entry += "\n"

        with open(self.changelog_path, 'a') as f:
            f.write(entry)

    def _check_nbconvert_available(self) -> bool:
        """
        Check if nbconvert is available.

        Returns:
            True if nbconvert is available, False otherwise
        """
        try:
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--version'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _verify_conversion_capability(self, to_python: bool = False, to_markdown: bool = False) -> bool:
        """
        Verify that conversion to the requested format is possible.

        Args:
            to_python: Check for Python conversion capability
            to_markdown: Check for Markdown conversion capability

        Returns:
            True if conversion is possible, False otherwise
        """
        # If copying .ipynb, no special tools needed
        if not to_python and not to_markdown:
            return True

        # Check if nbconvert is available
        if not self.nbconvert_available:
            print("\n⚠️  Warning: jupyter nbconvert not found")
            print("   Attempting to use fallback method (requires nbformat and nbconvert Python packages)")

            # Try to import the fallback libraries
            try:
                import importlib.util
                if (importlib.util.find_spec('nbformat') is not None
                        and importlib.util.find_spec('nbconvert') is not None):
                    print("   ✓ Fallback packages available")
                    return True
                raise ImportError("nbformat or nbconvert not found")
            except ImportError as e:
                print(f"   ✗ Fallback packages not available: {e}")
                print("\n💡 To fix this, install nbconvert:")
                print("   pip install nbconvert")
                print("   or")
                print("   conda install nbconvert")
                return False

        return True

    def list_notebooks(self) -> List[Path]:
        """
        List all available notebooks in source directory.

        Returns:
            List of notebook paths
        """
        notebooks = sorted(self.source_dir.glob("*.ipynb"))
        # Exclude checkpoint files
        notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]
        return notebooks

    def find_notebooks(self, pattern: Optional[str] = None) -> List[Path]:
        """
        Find notebooks matching a pattern.

        Args:
            pattern: Search pattern (case-insensitive substring match). If None, returns all.

        Returns:
            List of matching notebook paths
        """
        all_notebooks = self.list_notebooks()

        if pattern is None:
            return all_notebooks

        pattern_lower = pattern.lower()
        return [nb for nb in all_notebooks if pattern_lower in nb.name.lower()]

    def convert_notebooks(self, notebooks: List[Path], dry_run: bool = False,
                          to_python: bool = False, to_markdown: bool = False,
                          strip_outputs: bool = False) -> Dict[str, bool]:
        """
        Convert notebooks to dev directory.

        Args:
            notebooks: List of notebook paths to convert
            dry_run: If True, only show what would be converted without copying
            to_python: If True, convert .ipynb to .py format
            to_markdown: If True, convert .ipynb to .md format
            strip_outputs: If True, exclude output cells from markdown conversion

        Returns:
            Dictionary mapping filenames to success status
        """
        if not notebooks:
            print("⚠️  No notebooks to convert")
            return {}

        # Verify conversion capability before proceeding
        if not dry_run and not self._verify_conversion_capability(to_python, to_markdown):
            print("\n✗ Cannot proceed with conversion - missing required dependencies")
            return {nb.name: False for nb in notebooks}

        if not dry_run:
            self.ensure_dev_dir()

        results = {}
        converted_files = []

        for notebook in notebooks:
            try:
                if to_python:
                    # Convert to .py format
                    dest_path = self.dev_dir / notebook.stem
                    dest_path = dest_path.with_suffix('.py')

                    if dry_run:
                        status = "EXISTS" if dest_path.exists() else "NEW"
                        print(f"  [{status}] {notebook.name} -> {dest_path.name}")
                        results[dest_path.name] = True
                    else:
                        success = self._convert_to_python(notebook, dest_path)
                        if success:
                            print(f"✓ Converted: {notebook.name} -> dev/{dest_path.name}")
                            results[dest_path.name] = True
                            converted_files.append(dest_path.name)
                        else:
                            print(f"✗ Failed to convert {notebook.name} to Python")
                            results[dest_path.name] = False
                elif to_markdown:
                    # Convert to .md format
                    dest_path = self.dev_dir / notebook.stem
                    dest_path = dest_path.with_suffix('.md')

                    if dry_run:
                        status = "EXISTS" if dest_path.exists() else "NEW"
                        mode = " (no outputs)" if strip_outputs else ""
                        print(f"  [{status}] {notebook.name} -> {dest_path.name}{mode}")
                        results[dest_path.name] = True
                    else:
                        success = self._convert_to_markdown(notebook, dest_path,
                                                            strip_outputs=strip_outputs)
                        if success:
                            mode = " (no outputs)" if strip_outputs else ""
                            print(f"✓ Converted: {notebook.name} -> dev/{dest_path.name}{mode}")
                            results[dest_path.name] = True
                            converted_files.append(dest_path.name)
                        else:
                            print(f"✗ Failed to convert {notebook.name} to Markdown")
                            results[dest_path.name] = False
                else:
                    # Copy as .ipynb (existing behavior)
                    dest_path = self.dev_dir / notebook.name

                    if dry_run:
                        status = "EXISTS" if dest_path.exists() else "NEW"
                        print(f"  [{status}] {notebook.name}")
                        results[notebook.name] = True
                    else:
                        shutil.copy2(notebook, dest_path)
                        print(f"✓ Converted: {notebook.name} -> dev/{notebook.name}")
                        results[notebook.name] = True
                        converted_files.append(notebook.name)
            except Exception as e:
                # Determine the expected output filename for error reporting
                if to_python:
                    output_name = notebook.stem + '.py'
                elif to_markdown:
                    output_name = notebook.stem + '.md'
                else:
                    output_name = notebook.name
                print(f"✗ Failed to convert {notebook.name}: {e}")
                results[output_name] = False

        if not dry_run and converted_files:
            if to_python:
                operation = "Converted to Python"
            elif to_markdown:
                operation = "Converted to Markdown (no outputs)" if strip_outputs else "Converted to Markdown"
            else:
                operation = "Converted"
            self._append_changelog(operation, converted_files)

        return results

    def _convert_to_python(self, notebook_path: Path, output_path: Path) -> bool:
        """
        Convert a Jupyter notebook to Python script.

        Args:
            notebook_path: Path to source .ipynb file
            output_path: Path to destination .py file

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Use nbconvert via subprocess for reliability
            output_without_ext = str(output_path.with_suffix(''))

            result = subprocess.run(
                [
                    'jupyter', 'nbconvert',
                    '--to', 'python',
                    '--output', output_without_ext,
                    str(notebook_path.absolute())
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30  # Add timeout for safety
            )

            if result.returncode == 0:
                return True
            else:
                print(f"  nbconvert error: {result.stderr}")
                return False

        except FileNotFoundError:
            # Fallback: try using nbconvert Python API
            print("  jupyter command not found, trying Python API fallback...")
            try:
                import nbformat
                from nbconvert import PythonExporter

                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)

                exporter = PythonExporter()
                python_code, _ = exporter.from_notebook_node(notebook)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(python_code)

                return True
            except ImportError as e:
                print(f"  Fallback unavailable: {e}")
                print("  Install with: pip install nbformat nbconvert")
                return False
            except Exception as e:
                print(f"  Fallback conversion error: {e}")
                return False
        except subprocess.TimeoutExpired:
            print("  Conversion timeout after 30s")
            return False
        except PermissionError as e:
            print(f"  Permission denied: {e}")
            return False
        except OSError as e:
            print(f"  OS error: {e}")
            return False

    def _convert_to_markdown(self, notebook_path: Path, output_path: Path,
                             strip_outputs: bool = False) -> bool:
        """
        Convert a Jupyter notebook to Markdown.

        Args:
            notebook_path: Path to source .ipynb file
            output_path: Path to destination .md file
            strip_outputs: If True, exclude output cells from conversion

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Use nbconvert via subprocess for reliability
            # Note: --output expects path without extension
            output_without_ext = str(output_path.with_suffix(''))

            # Base command
            cmd = [
                'jupyter', 'nbconvert',
                '--to', 'markdown',
                '--output', output_without_ext,
            ]

            # Add output stripping flags if requested
            if strip_outputs:
                cmd.extend([
                    '--no-prompt',
                    '--TemplateExporter.exclude_output=True',
                    '--TemplateExporter.exclude_output_prompt=True',
                ])

            # Add input file
            cmd.append(str(notebook_path.absolute()))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30
            )

            if result.returncode == 0:
                return True
            else:
                print(f"  nbconvert error: {result.stderr}")
                return False

        except FileNotFoundError:
            # Fallback: try using nbconvert Python API
            try:
                import nbformat
                from nbconvert import MarkdownExporter
                from traitlets.config import Config

                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)

                # Configure exporter to strip outputs if requested
                if strip_outputs:
                    c = Config()
                    c.TemplateExporter.exclude_output = True
                    c.TemplateExporter.exclude_output_prompt = True
                    c.TemplateExporter.exclude_input_prompt = True
                    exporter = MarkdownExporter(config=c)
                else:
                    exporter = MarkdownExporter()

                markdown_content, resources = exporter.from_notebook_node(notebook)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                # Handle image files if present
                if 'outputs' in resources and not strip_outputs:
                    images_dir = output_path.parent / f"{output_path.stem}_files"
                    images_dir.mkdir(exist_ok=True)
                    for filename, data in resources['outputs'].items():
                        image_path = images_dir / filename
                        with open(image_path, 'wb') as img_file:
                            img_file.write(data)

                return True
            except Exception as e:
                print(f"  Fallback conversion error: {e}")
                return False
        except subprocess.TimeoutExpired:
            print("  Conversion timeout after 30s")
            return False
        except PermissionError as e:
            print(f"  Permission denied: {e}")
            return False
        except OSError as e:
            print(f"  OS error: {e}")
            return False
        except Exception as e:
            print(f"  Conversion error: {e}")
            return False

    def detect_dev_file_format(self, notebook_stem: str) -> Optional[Tuple[Path, FileFormat]]:
        """
        Detect what format exists in dev directory for given notebook.

        Args:
            notebook_stem: Notebook name without extension

        Returns:
            Tuple of (dev_file_path, format) or None if not found
        """
        for fmt in FileFormat:
            dev_file = self.dev_dir / f"{notebook_stem}{fmt.value}"
            if dev_file.exists():
                return (dev_file, fmt)
        return None

    def find_sync_pairs(self, pattern: Optional[str] = None) -> List[SyncPair]:
        """
        Find all sync pairs (source .ipynb + dev file in any format).

        Args:
            pattern: Optional pattern to filter notebooks

        Returns:
            List of SyncPair objects
        """
        # Get all source notebooks
        source_notebooks = self.find_notebooks(pattern)

        # Also check dev directory for orphaned files
        dev_files = {}
        if self.dev_dir.exists():
            for ext in ['.ipynb', '.py', '.md']:
                for f in self.dev_dir.glob(f"*{ext}"):
                    if f.stem not in ['README', 'NB_CHANGELOG']:
                        dev_files[f.stem] = f

        # Build pairs
        pairs = []
        all_stems = set([nb.stem for nb in source_notebooks] + list(dev_files.keys()))

        if pattern:
            pattern_lower = pattern.lower()
            all_stems = {s for s in all_stems if pattern_lower in s.lower()}

        for stem in sorted(all_stems):
            source_ipynb = self.source_dir / f"{stem}.ipynb"
            dev_info = self.detect_dev_file_format(stem)

            if dev_info:
                dev_file, fmt = dev_info
                pairs.append(SyncPair(
                    notebook_stem=stem,
                    source_ipynb=source_ipynb,
                    dev_file=dev_file,
                    format=fmt
                ))

        return pairs

    def _convert_to_temp(self, notebook_path: Path, target_format: FileFormat,
                         strip_outputs: bool = True) -> Optional[Path]:
        """
        Convert notebook to temporary file in target format.

        Args:
            notebook_path: Source .ipynb file
            target_format: Target format
            strip_outputs: Whether to strip outputs (for markdown)

        Returns:
            Path to temporary file or None on failure
        """
        # Create temp file with appropriate extension
        temp_fd, temp_path = tempfile.mkstemp(suffix=target_format.value)
        temp_path = Path(temp_path)

        os.close(temp_fd)

        try:
            if target_format == FileFormat.IPYNB:
                # Just copy
                shutil.copy2(notebook_path, temp_path)
                return temp_path

            elif target_format == FileFormat.PYTHON:
                success = self._convert_to_python(notebook_path, temp_path)
                return temp_path if success else None

            elif target_format == FileFormat.MARKDOWN:
                success = self._convert_to_markdown(notebook_path, temp_path,
                                                    strip_outputs=strip_outputs)
                return temp_path if success else None

        except Exception as e:
            print(f"  Conversion error: {e}")
            if temp_path.exists():
                temp_path.unlink()
            return None

    def compare_files(self, file1: Path, file2: Path) -> Tuple[bool, List[str]]:
        """
        Compare two text files and return diff.

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            Tuple of (files_identical, diff_lines)
        """
        try:
            with open(file1, 'r', encoding='utf-8') as f:
                lines1 = f.readlines()
            with open(file2, 'r', encoding='utf-8') as f:
                lines2 = f.readlines()

            # Check if identical
            if lines1 == lines2:
                return True, []

            # Generate unified diff
            diff = list(difflib.unified_diff(
                lines1, lines2,
                fromfile=f"source/{file1.name}",
                tofile=f"dev/{file2.name}",
                lineterm=''
            ))

            return False, diff

        except Exception as e:
            return False, [f"Error comparing files: {e}"]

    def sync_notebooks(self, sync_pair: SyncPair, show_diff: bool = True,
                       max_diff_lines: int = 50) -> Optional[bool]:
        """
        Sync a pair of notebooks by converting source to dev format and comparing.

        Args:
            sync_pair: SyncPair object with source and dev paths
            show_diff: Whether to show diff output
            max_diff_lines: Maximum lines of diff to show

        Returns:
            True if files are identical, False if different, None on error
        """
        print(f"\n📊 {sync_pair.notebook_stem}{sync_pair.format.value}")
        print("=" * 60)

        # Check existence
        if not sync_pair.source_exists:
            print("  ⚠️  Source notebook not found (orphaned dev file)")
            return None

        if not sync_pair.dev_exists:
            print("  ⚠️  Dev file not found")
            return None

        # Convert source to same format as dev
        print(f"  Converting source to {sync_pair.format.value} for comparison...")
        temp_source = self._convert_to_temp(
            sync_pair.source_ipynb,
            sync_pair.format,
            strip_outputs=True  # Always strip outputs for fair comparison
        )

        if not temp_source:
            print("  ✗ Failed to convert source notebook")
            return None

        try:
            # Compare files
            identical, diff_lines = self.compare_files(temp_source, sync_pair.dev_file)

            if identical:
                print("  ✓ Files are identical")
                return True

            # Show diff
            print(f"  ⚠️  Files differ ({len(diff_lines)} diff lines)")

            if show_diff and diff_lines:
                print("\n  Diff preview:")
                print("  " + "-" * 58)
                for line in diff_lines[:max_diff_lines]:
                    # Color code diff lines
                    if line.startswith('+++') or line.startswith('---'):
                        print(f"  {line}")
                    elif line.startswith('+'):
                        print(f"  + {line[1:]}")
                    elif line.startswith('-'):
                        print(f"  - {line[1:]}")
                    elif line.startswith('@@'):
                        print(f"  {line}")

                if len(diff_lines) > max_diff_lines:
                    print(f"  ... ({len(diff_lines) - max_diff_lines} more lines)")

            return False

        finally:
            # Cleanup temp file
            if temp_source and temp_source.exists():
                temp_source.unlink()

    def determine_merge_direction(self, sync_pair: SyncPair) -> Optional[MergeDirection]:
        """
        Determine which direction to merge based on file timestamps.

        Args:
            sync_pair: SyncPair object with file paths

        Returns:
            Recommended merge direction or None if error
        """
        if not sync_pair.source_exists or not sync_pair.dev_exists:
            return None

        source_mtime = sync_pair.source_ipynb.stat().st_mtime
        dev_mtime = sync_pair.dev_file.stat().st_mtime

        if dev_mtime > source_mtime:
            return MergeDirection.DEV_TO_SOURCE
        elif source_mtime > dev_mtime:
            return MergeDirection.SOURCE_TO_DEV
        else:
            # Same timestamp - no clear winner
            return None

    def merge_files(self, sync_pair: SyncPair, direction: MergeDirection,
                    dry_run: bool = False, force: bool = False) -> bool:
        """
        Merge files in specified direction.

        Args:
            sync_pair: SyncPair object
            direction: Direction to merge (dev→source or source→dev)
            dry_run: If True, only show what would be done
            force: If True, skip timestamp check

        Returns:
            True if merge successful or would succeed
        """
        # Auto-detect if requested
        if direction == MergeDirection.AUTO:
            detected = self.determine_merge_direction(sync_pair)
            if not detected and not force:
                print("  ⚠️  Cannot auto-detect merge direction (same timestamp)")
                print("     Use --merge-direction to specify explicitly")
                return False
            direction = detected

        if direction == MergeDirection.DEV_TO_SOURCE:
            return self._merge_dev_to_source(sync_pair, dry_run)
        elif direction == MergeDirection.SOURCE_TO_DEV:
            return self._merge_source_to_dev(sync_pair, dry_run)

        return False

    def _merge_dev_to_source(self, sync_pair: SyncPair, dry_run: bool = False) -> bool:
        """
        Merge dev file to source (original behavior).

        Args:
            sync_pair: SyncPair object
            dry_run: If True, only show what would be done

        Returns:
            True if merge successful
        """
        if sync_pair.format == FileFormat.IPYNB:
            # Direct copy for .ipynb
            if dry_run:
                print(f"  Would copy: dev/{sync_pair.dev_file.name} → {sync_pair.source_ipynb.name}")
                return True

            try:
                shutil.copy2(sync_pair.dev_file, sync_pair.source_ipynb)
                print(f"  ✓ Merged: dev/{sync_pair.dev_file.name} → {sync_pair.source_ipynb.name}")
                self._append_changelog("Merged dev to source", [sync_pair.source_ipynb.name])
                return True
            except Exception as e:
                print(f"  ✗ Failed to merge: {e}")
                return False

        else:
            # Manual merge required for .py/.md
            print(f"  ℹ️  Manual merge required for {sync_pair.format.value} files")
            print("     Dev file has changes that need review:")
            print(f"     1. Review: dev/{sync_pair.dev_file.name}")
            print(f"     2. Update: {sync_pair.source_ipynb.name}")
            print(
                f"     3. Reconvert: /dev:nb-convert --notebook-name '{sync_pair.notebook_stem}' {self._format_to_flag(sync_pair.format)}")
            return False

    def _merge_source_to_dev(self, sync_pair: SyncPair, dry_run: bool = False) -> bool:
        """
        Merge source to dev (refresh dev copy).

        Args:
            sync_pair: SyncPair object
            dry_run: If True, only show what would be done

        Returns:
            True if merge successful
        """
        if sync_pair.format == FileFormat.IPYNB:
            # Direct copy for .ipynb
            if dry_run:
                print(f"  Would copy: {sync_pair.source_ipynb.name} → dev/{sync_pair.dev_file.name}")
                return True

            try:
                shutil.copy2(sync_pair.source_ipynb, sync_pair.dev_file)
                print(f"  ✓ Refreshed: {sync_pair.source_ipynb.name} → dev/{sync_pair.dev_file.name}")
                self._append_changelog("Refreshed dev from source", [sync_pair.dev_file.name])
                return True
            except Exception as e:
                print(f"  ✗ Failed to refresh: {e}")
                return False

        else:
            # Need to reconvert for .py/.md
            if dry_run:
                print(f"  Would reconvert: {sync_pair.source_ipynb.name} → dev/{sync_pair.dev_file.name}")
                return True

            print(f"  🔄 Reconverting source to {sync_pair.format.value}...")

            # Determine conversion parameters
            to_python = sync_pair.format == FileFormat.PYTHON
            to_markdown = sync_pair.format == FileFormat.MARKDOWN
            strip_outputs = True  # Always strip for consistency

            # Convert
            if to_python:
                success = self._convert_to_python(sync_pair.source_ipynb, sync_pair.dev_file)
            elif to_markdown:
                success = self._convert_to_markdown(sync_pair.source_ipynb, sync_pair.dev_file,
                                                    strip_outputs=strip_outputs)
            else:
                success = False

            if success:
                print(f"  ✓ Refreshed: {sync_pair.source_ipynb.name} → dev/{sync_pair.dev_file.name}")
                self._append_changelog("Refreshed dev from source (reconverted)", [sync_pair.dev_file.name])
                return True
            else:
                print("  ✗ Failed to reconvert")
                return False

    def _format_to_flag(self, fmt: FileFormat) -> str:
        """Convert FileFormat to CLI flag."""
        if fmt == FileFormat.PYTHON:
            return '--to-python'
        elif fmt == FileFormat.MARKDOWN:
            return '--to-markdown --strip-outputs'
        return ''

    def cleanup_dev_dir(self, confirm: bool = False) -> bool:
        """
        Remove dev directory and all its contents.

        Args:
            confirm: If True, skip confirmation prompt

        Returns:
            True if cleanup successful, False otherwise
        """
        if not self.dev_dir.exists():
            print("ℹ️  Dev directory doesn't exist, nothing to clean up")
            return True

        # List what will be deleted
        notebooks = list(self.dev_dir.glob("*.ipynb"))
        python_files = list(self.dev_dir.glob("*.py"))
        md_files = list(self.dev_dir.glob("*.md"))
        all_files = notebooks + python_files + md_files

        print(f"\n⚠️  This will DELETE {len(all_files)} file(s) and the dev directory:")
        for f in all_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        if len(all_files) > 5:
            print(f"  ... and {len(all_files) - 5} more")

        if not confirm:
            response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("✗ Cleanup cancelled")
                return False

        try:
            shutil.rmtree(self.dev_dir)
            print(f"✓ Removed dev directory: {self.dev_dir}")
            return True
        except Exception as e:
            print(f"✗ Failed to remove dev directory: {e}")
            return False


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to dev directory for testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all notebooks to .ipynb
  python nb_convert.py --all

  # Convert all notebooks to .py format
  python nb_convert.py --all --to-python

  # Convert all notebooks to .md format (with outputs)
  python nb_convert.py --all --to-markdown

  # Convert all notebooks to .md format (without outputs)
  python nb_convert.py --all --to-markdown --strip-outputs

  # Convert specific notebook(s) to .md by pattern (no outputs)
  python nb_convert.py --notebook-name "graph_PostGIS" --to-markdown --strip-outputs

  # List all available notebooks
  python nb_convert.py --list

  # Clean up dev directory
  python nb_convert.py --cleanup
        """
    )

    # Mutually exclusive operation flags
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--all', action='store_true',
                           help='Convert all notebooks to dev directory')
    operation.add_argument('--notebook-name', type=str, metavar='PATTERN',
                           help='Convert notebook(s) matching pattern (case-insensitive)')
    operation.add_argument('--cleanup', action='store_true',
                           help='Remove dev directory and all converted notebooks')
    operation.add_argument('--check', action='store_true',
                           help='Show what would be converted without actually converting (dry run)')
    operation.add_argument('--list', action='store_true',
                           help='List all available notebooks in source directory')
    operation.add_argument('--sync', action='store_true',
                           help='Compare dev files with source notebooks (can be combined with --notebook-name)')

    # Format options (mutually exclusive)
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('--to-python', action='store_true',
                              help='Convert notebooks to .py format instead of copying .ipynb')
    format_group.add_argument('--to-markdown', action='store_true',
                              help='Convert notebooks to .md format instead of copying .ipynb')

    # Additional options
    parser.add_argument('--strip-outputs', action='store_true',
                        help='Strip output cells when converting to Markdown (ignored for other formats)')
    parser.add_argument('--project-root', type=Path,
                        help='Project root directory (auto-detected if not specified)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts (use with --cleanup)')


    # Sync-specific options
    parser.add_argument('--merge', action='store_true',
                        help='Merge changes between source and dev')
    parser.add_argument('--merge-direction',
                        choices=['dev-to-source', 'source-to-dev', 'auto'],
                        default='auto',
                        help='Merge direction (default: auto-detect from timestamps)')
    parser.add_argument('--force-merge', action='store_true',
                        help='Force merge even if timestamps suggest otherwise')
    parser.add_argument('--show-diff', action='store_true', default=True,
                        help='Show diff output (use with --sync)')
    parser.add_argument('--max-diff-lines', type=int, default=50,
                        help='Maximum diff lines to show (default: 50)')

    args = parser.parse_args()

    # Initialize converter
    converter = NotebookConverter(project_root=args.project_root)

    # Execute requested operation
    if args.list:
        # List all notebooks
        notebooks = converter.list_notebooks()
        print(f"\n📒 Available notebooks in {converter.source_dir}:\n")
        for i, nb in enumerate(notebooks, 1):
            print(f"  {i}. {nb.name}")
        print(f"\nTotal: {len(notebooks)} notebooks")
        return 0

    elif args.cleanup:
        # Clean up dev directory
        success = converter.cleanup_dev_dir(confirm=args.yes)
        return 0 if success else 1

    elif args.check:
        # Dry run - show what would be converted
        print("\n🔍 Checking notebooks (dry run)...\n")
        pattern = input("Enter notebook name pattern (or press Enter for all): ").strip()
        pattern = pattern if pattern else None

        notebooks = converter.find_notebooks(pattern)

        if not notebooks:
            print(f"⚠️  No notebooks found matching: {pattern}")
            return 1

        if args.to_python:
            format_msg = "to Python scripts"
        elif args.to_markdown:
            format_msg = "to Markdown"
        else:
            format_msg = "as notebooks"

        print(f"\n📋 Would convert {len(notebooks)} notebook(s) {format_msg}:\n")
        converter.convert_notebooks(notebooks, dry_run=True,
                                    to_python=args.to_python,
                                    to_markdown=args.to_markdown)
        return 0


    elif args.sync:
        # Sync operation with conversion-based comparison
        pattern = args.notebook_name if hasattr(args, 'notebook_name') and args.notebook_name else None
        pairs = converter.find_sync_pairs(pattern)

        if not pairs:
            print("⚠️  No sync pairs found")
            return 1

        print("\n🔄 Notebook Sync Report (Conversion-Based Comparison)")
        print("=" * 60)

        results = []
        for pair in pairs:
            result = converter.sync_notebooks(pair, show_diff=args.show_diff,
                                              max_diff_lines=args.max_diff_lines)
            results.append((pair, result))

        # Merge if requested
        if args.merge:
            print("\n" + "=" * 60)
            direction = MergeDirection(args.merge_direction)
            print(f"🔀 Merge Operation ({direction.value})")
            print("=" * 60)
            for pair, identical in results:
                if identical is True:  # Explicit check
                    print(f"\n  ⏭️  Skipping {pair.notebook_stem} (no changes)")
                elif identical is None:
                    print(f"\n  ⚠️  Skipping {pair.notebook_stem} (error)")
                else: # identical is False
                    # Show recommended direction
                    recommended = converter.determine_merge_direction(pair)
                    if recommended and direction == MergeDirection.AUTO:
                        print(f"\n  Processing: {pair.notebook_stem}")
                        print(f"  📊 Recommended: {recommended.value}")
                    else:
                        print(f"\n  Processing: {pair.notebook_stem}")
                    if not args.yes:
                        merge_dir_display = direction.value if direction != MergeDirection.AUTO else recommended.value
                        response = input(f"    Merge {merge_dir_display}? (y/n): ")
                        if response.lower() not in ['y', 'yes']:
                            print("    Skipped")
                            continue
                    converter.merge_files(pair, direction,
                                          dry_run=args.check,
                                          force=args.force_merge)
        return 0

    elif args.all or args.notebook_name:
        # Convert notebooks
        if args.all:
            notebooks = converter.find_notebooks(None)
            if args.to_python:
                format_msg = "to Python scripts"
            elif args.to_markdown:
                format_msg = "to Markdown"
            else:
                format_msg = ""
            print(f"\n📝 Converting ALL {len(notebooks)} notebooks {format_msg}...\n")
        else:
            notebooks = converter.find_notebooks(args.notebook_name)
            if not notebooks:
                print(f"\n⚠️  No notebooks found matching pattern: '{args.notebook_name}'")
                print("\nUse --list to see all available notebooks")
                return 1
            if args.to_python:
                format_msg = "to Python scripts"
            elif args.to_markdown:
                format_msg = "to Markdown"
            else:
                format_msg = ""
            print(f"\n📝 Converting {len(notebooks)} notebook(s) {format_msg} matching '{args.notebook_name}'...\n")

        # Show files before conversion
        print("Files to convert:")
        for nb in notebooks:
            if args.to_python:
                output_name = nb.stem + '.py'
            elif args.to_markdown:
                output_name = nb.stem + '.md'
            else:
                output_name = nb.name
            print(f"  - {nb.name} -> {output_name}")
        print()

        results = converter.convert_notebooks(notebooks,
                                              to_python=args.to_python,
                                              to_markdown=args.to_markdown,
                                              strip_outputs=args.strip_outputs)

        # Summary
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)

        print(f"\n📊 Summary: {success_count}/{total_count} notebooks converted successfully")

        if success_count < total_count:
            print("\n❌ Some conversions failed:")
            for fname, success in results.items():
                if not success:
                    print(f"  - {fname}")
            return 1

        return 0

    return 0

if __name__ == "__main__":
    sys.exit(main())