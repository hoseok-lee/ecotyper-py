"""Sphinx configuration for the ecotyper (Python port) documentation.

Build:  pip install -e ".[docs]" && sphinx-build -b html docs docs/_build/html
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE / "_scripts"))

import make_benchmark_page  # noqa: E402
import make_cli_help  # noqa: E402

# docs/_generated/: benchmark charts and tables (from validation/bench, validation/reports) and CLI help
make_benchmark_page.build()
make_cli_help.build()

project = "ecotyper"
author = "Python port contributors (EcoTyper: Luca, Steen, Newman et al.)"
copyright = "Stanford University (EcoTyper); non-commercial license"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "_generated/*.md", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": True}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["icu"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
}

html_theme = "furo"
html_title = "ecotyper (Python port)"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/hoseok-lee/ecotyper-py",
    "source_branch": "main",
    "source_directory": "docs/",
}
copybutton_prompt_text = r"\$ |>>> "
copybutton_prompt_is_regexp = True
