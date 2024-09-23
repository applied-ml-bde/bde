"""Documentation configuration file for the project."""

import os
import sys

project = 'BDE'
copyright = '2024, Konstantin and Steffen'
author = 'Konstantin and Steffen'
release = '1.0.0'

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv', '.github']

# sys.path.append(os.path.abspath('../src'))

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/logo.png'
html_theme_options = {
    'logo_only': False,
    'display_version': False,
}

# autodoc
autodoc_typehints = 'description'
