#!/usr/bin/env python3
"""
Wiki Generator for nBedR Documentation

This script processes documentation files from the repository and generates
GitHub Wiki pages with proper navigation, cross-references, and formatting.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class WikiGenerator:
    """Generates GitHub Wiki pages from repository documentation."""

    def __init__(self, config_path: str = "wiki-config.yaml", repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.wiki_dir = self.repo_root / "wiki"

        # Track changes for incremental updates
        self.changes_detected = False
        self.content_hashes = {}

    def _load_config(self) -> dict:
        """Load wiki configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _get_timestamp(self) -> str:
        """Get current timestamp for documentation."""
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for change detection."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_existing_hashes(self) -> dict:
        """Load existing content hashes for change detection."""
        hash_file = self.wiki_dir / ".content-hashes.json"
        if hash_file.exists():
            with open(hash_file, "r") as f:
                return json.load(f)
        return {}

    def _save_content_hashes(self):
        """Save content hashes for future change detection."""
        hash_file = self.wiki_dir / ".content-hashes.json"
        with open(hash_file, "w") as f:
            json.dump(self.content_hashes, f, indent=2)

    def _process_links(self, content: str, source_file: str) -> str:
        """Convert relative links to wiki links."""
        if not self.config["wiki"]["processing"]["convert_links"]:
            return content

        # Convert markdown links: [text](path.md) -> [text](Wiki-Page)
        def link_replacer(match):
            text = match.group(1)
            url = match.group(2)

            # Skip external links
            if url.startswith(("http://", "https://", "mailto:")):
                return match.group(0)

            # Skip anchors
            if url.startswith("#"):
                return match.group(0)

            # Convert local markdown files to wiki pages
            if url.endswith(".md"):
                # Find corresponding wiki page
                for source, wiki_page in self.config["wiki"]["pages"].items():
                    if source.endswith(url) or url.endswith(source):
                        wiki_name = wiki_page.replace(".md", "")
                        return f"[{text}]({wiki_name})"

                # Default conversion: remove .md and convert to title case
                wiki_name = Path(url).stem.replace("_", "-").replace(" ", "-")
                return f"[{text}]({wiki_name})"

            return match.group(0)

        # Process markdown links
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replacer, content)

        return content

    def _generate_toc(self, content: str) -> str:
        """Generate table of contents for long pages."""
        if len(content) < self.config["wiki"]["processing"]["toc_min_length"]:
            return ""

        max_level = self.config["wiki"]["processing"]["max_toc_level"]

        # Extract headings
        headings = []
        for match in re.finditer(r"^(#{1," + str(max_level) + r"})\s+(.+)$", content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            anchor = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "-").lower()
            headings.append((level, title, anchor))

        if len(headings) < 3:  # Don't generate TOC for short content
            return ""

        # Generate TOC markdown
        toc_lines = ["## Table of Contents", ""]
        for level, title, anchor in headings:
            indent = "  " * (level - 1)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")

        toc_lines.extend(["", "---", ""])
        return "\n".join(toc_lines)

    def _process_mermaid(self, content: str) -> str:
        """Process Mermaid diagrams for wiki compatibility."""
        if not self.config["wiki"]["processing"]["process_mermaid"]:
            return content

        # GitHub Wiki supports Mermaid, but let's ensure proper formatting
        def mermaid_replacer(match):
            diagram_type = match.group(1) if match.group(1) else "graph"
            diagram_content = match.group(2)

            # Ensure proper mermaid code block format
            return f"```mermaid\n{diagram_type}\n{diagram_content}\n```"

        # Match mermaid blocks
        content = re.sub(r"```mermaid\s*\n?([a-zA-Z]+)?\s*\n(.*?)\n```", mermaid_replacer, content, flags=re.DOTALL)

        return content

    def _consolidate_deployment_docs(self) -> str:
        """Consolidate deployment documentation from deployment/ directory."""
        deployment_dir = self.repo_root / "deployment"
        if not deployment_dir.exists():
            return "# Deployment Guide\n\nNo deployment documentation found."

        content = ["# Deployment Guide", ""]
        content.append("This guide covers deployment configurations and procedures for nBedR.")
        content.append("")

        # Add main deployment README
        readme_path = deployment_dir / "README.md"
        if readme_path.exists():
            with open(readme_path, "r") as f:
                readme_content = f.read()
                # Remove the title if it exists
                readme_content = re.sub(r"^#\s+.*\n", "", readme_content)
                content.append(readme_content)

        # Add Docker documentation
        docker_dir = deployment_dir / "docker"
        if docker_dir.exists():
            content.extend(["", "## Docker Deployment", ""])

            docker_readme = docker_dir / "README.md"
            if docker_readme.exists():
                with open(docker_readme, "r") as f:
                    docker_content = f.read()
                    docker_content = re.sub(r"^#\s+.*\n", "", docker_content)
                    content.append(docker_content)

            # Document Dockerfiles
            for dockerfile in docker_dir.glob("Dockerfile*"):
                content.extend(["", f"### {dockerfile.name}", ""])
                content.append(f"Location: `{dockerfile.relative_to(self.repo_root)}`")
                content.append("")

        # Add Kubernetes documentation
        k8s_dir = deployment_dir / "kubernetes"
        if k8s_dir.exists():
            content.extend(["", "## Kubernetes Deployment", ""])
            content.append("The following Kubernetes configurations are available:")
            content.append("")

            for cloud_dir in k8s_dir.iterdir():
                if cloud_dir.is_dir() and cloud_dir.name != "base":
                    content.append(f"### {cloud_dir.name.upper()}")
                    content.append(f"Configuration files in `{cloud_dir.relative_to(self.repo_root)}/`")
                    content.append("")

        # Add scripts documentation
        scripts_dir = deployment_dir / "scripts"
        if scripts_dir.exists():
            content.extend(["", "## Deployment Scripts", ""])

            for script in scripts_dir.glob("*.sh"):
                content.append(f"- **{script.name}**: {script.relative_to(self.repo_root)}")
            content.append("")

        return "\n".join(content)

    def _consolidate_templates(self) -> str:
        """Consolidate template documentation."""
        templates_dir = self.repo_root / "templates"
        if not templates_dir.exists():
            return "# Templates\n\nNo templates found."

        content = ["# Templates", ""]
        content.append("This page documents the available templates and their usage.")
        content.append("")

        # Add main templates README if it exists
        readme_path = templates_dir / "README.md"
        if readme_path.exists():
            with open(readme_path, "r") as f:
                readme_content = f.read()
                readme_content = re.sub(r"^#\s+.*\n", "", readme_content)
                content.append(readme_content)

        # List available templates
        content.extend(["", "## Available Templates", ""])

        for template_file in templates_dir.glob("*.txt"):
            content.append(f"### {template_file.stem.replace('_', ' ').title()}")
            content.append(f"File: `{template_file.relative_to(self.repo_root)}`")
            content.append("")

            # Add first few lines as preview
            with open(template_file, "r") as f:
                lines = f.readlines()[:5]
                if lines:
                    content.append("**Preview:**")
                    content.append("```")
                    content.extend([line.rstrip() for line in lines])
                    content.append("```")
                    content.append("")

        return "\n".join(content)

    def _generate_sidebar(self) -> str:
        """Generate navigation sidebar for the wiki."""
        sidebar_content = ["# Navigation", ""]

        navigation = self.config["wiki"]["navigation"]
        for section in navigation:
            sidebar_content.append(f"## {section['title']}")
            sidebar_content.append("")

            for page in section["pages"]:
                sidebar_content.append(f"- [{page.replace('-', ' ')}]({page})")

            sidebar_content.append("")

        return "\n".join(sidebar_content)

    def _process_file(self, source_path: str, wiki_page: str) -> Tuple[str, bool]:
        """Process a single documentation file."""
        source_file = self.repo_root / source_path

        # Handle special consolidation cases
        if source_path == "deployment/":
            content = self._consolidate_deployment_docs()
        elif source_path == "templates/":
            content = self._consolidate_templates()
        elif source_file.exists():
            with open(source_file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            print(f"Warning: Source file not found: {source_file}")
            return "", False

        # Process content
        processed_content = self._process_links(content, source_path)
        processed_content = self._process_mermaid(processed_content)

        # Add table of contents if needed
        toc = self._generate_toc(processed_content)
        if toc:
            # Insert TOC after the first heading
            lines = processed_content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    lines.insert(i + 1, "\n" + toc)
                    break
            processed_content = "\n".join(lines)

        # Add header and footer
        timestamp = self._get_timestamp()
        header = self.config["wiki"]["header_template"].format(timestamp=timestamp, source_file=source_path)
        footer = self.config["wiki"]["footer_template"].format(timestamp=timestamp)

        final_content = header + processed_content + footer

        # Check for changes
        content_hash = self._calculate_content_hash(final_content)
        existing_hashes = self._load_existing_hashes()

        changed = wiki_page not in existing_hashes or existing_hashes[wiki_page] != content_hash

        if changed:
            self.content_hashes[wiki_page] = content_hash
            self.changes_detected = True

        return final_content, changed

    def generate_wiki(self, force: bool = False) -> bool:
        """Generate all wiki pages."""
        print(f"Generating wiki documentation from {self.repo_root}")

        # Create wiki directory
        self.wiki_dir.mkdir(exist_ok=True)

        # Load existing hashes for change detection
        existing_hashes = self._load_existing_hashes()

        pages_updated = []

        # Process configured pages
        for source_path, wiki_page in self.config["wiki"]["pages"].items():
            print(f"Processing {source_path} -> {wiki_page}")

            content, changed = self._process_file(source_path, wiki_page)

            if changed or force:
                wiki_file = self.wiki_dir / wiki_page
                with open(wiki_file, "w", encoding="utf-8") as f:
                    f.write(content)

                pages_updated.append(wiki_page)
                print(f"  ✓ Updated {wiki_page}")
            else:
                print(f"  - No changes in {wiki_page}")

        # Generate sidebar
        sidebar_content = self._generate_sidebar()
        sidebar_hash = self._calculate_content_hash(sidebar_content)

        if "_Sidebar.md" not in existing_hashes or existing_hashes["_Sidebar.md"] != sidebar_hash or force:

            sidebar_file = self.wiki_dir / "_Sidebar.md"
            with open(sidebar_file, "w", encoding="utf-8") as f:
                f.write(sidebar_content)

            self.content_hashes["_Sidebar.md"] = sidebar_hash
            pages_updated.append("_Sidebar.md")
            print("  ✓ Updated _Sidebar.md")

        # Save content hashes
        if self.changes_detected or force:
            self._save_content_hashes()

        if pages_updated:
            print(f"\n✅ Wiki generation complete! Updated {len(pages_updated)} pages:")
            for page in pages_updated:
                print(f"   - {page}")
            return True
        else:
            print("\n📄 No changes detected, wiki is up to date.")
            return False


def main():
    """Main entry point for the wiki generator."""
    parser = argparse.ArgumentParser(description="Generate GitHub Wiki from repository documentation")
    parser.add_argument("--config", default="wiki-config.yaml", help="Path to wiki configuration file")
    parser.add_argument("--repo-root", default=".", help="Repository root directory")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all pages")
    parser.add_argument(
        "--check-changes",
        action="store_true",
        help="Only check if changes are needed (exit code 0=no changes, 1=changes needed)",
    )

    args = parser.parse_args()

    try:
        generator = WikiGenerator(args.config, args.repo_root)

        if args.check_changes:
            # Just check for changes without generating
            for source_path, wiki_page in generator.config["wiki"]["pages"].items():
                _, changed = generator._process_file(source_path, wiki_page)
                if changed:
                    print("Changes detected in documentation")
                    exit(1)
            print("No changes detected")
            exit(0)

        has_changes = generator.generate_wiki(force=args.force)

        if has_changes:
            print("\n🚀 Wiki is ready for publishing!")

    except Exception as e:
        print(f"❌ Error generating wiki: {e}")
        exit(1)


if __name__ == "__main__":
    main()
