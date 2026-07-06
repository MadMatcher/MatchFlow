# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os, sys
sys.path.insert(0, os.path.abspath('../..'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MatchFlow'
copyright = '2025, Dev Ahluwalia, Derek Paulsen'
author = 'Dev Ahluwalia, Derek Paulsen'
# Version comes from the publish pipeline (docs/publish.sh sets DOCS_VERSION from
# the package's pyproject.toml) so the displayed + canonical version stays in sync
# with the URL path /matchflow/<version>/.
release = os.environ.get("DOCS_VERSION", "0.1.0")
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'imported-members': True,
    'special-members': '__all__',
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# HTML output: Furo, MadMatcher brand (mirrors madmatcher-pro/docs/conf.py so the
# whole docs.madmatcher.ai site looks like one product).
html_baseurl = f"https://docs.madmatcher.ai/matchflow/{release}/"
html_theme = 'furo'
html_static_path = ['_static']
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;560;680;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
    "custom.css",
]
html_title = "MatchFlow API reference"
html_logo = "_static/logo-mark.svg"
html_favicon = "_static/logo-mark.svg"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#c5050c",
        "color-brand-content": "#c5050c",
        "color-foreground-primary": "#1d1d1f",
        "color-background-primary": "#ffffff",
        "color-background-secondary": "#f6f6f7",
    },
    "dark_css_variables": {
        "color-brand-primary": "#c5050c",
        "color-brand-content": "#c5050c",
        "color-background-primary": "#161617",
        "color-background-secondary": "#1f1f21",
    },
}
