"""Documentation configuration file for the project."""

import os
import sys

project = 'BDE'
copyright = '2024, Konstantin and Steffen'
author = 'Konstantin and Steffen'
# release = get_version('bde')
release = '0.1.0'
version = ".".join(release.split(".")[:3])

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx-prompt",
    "sphinx_gallery.gen_gallery",
    "numpydoc",
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    "_templates",
    'Thumbs.db',
    '.DS_Store',
    '.venv',
    '.github',
]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# sys.path.append(os.path.abspath('../src'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_style = "css/project-template.css"
html_logo = "_static/img/logo.svg"
# html_favicon = ""
html_css_files = [
    "css/project-template.css",
]
html_sidebars: dict = {
    "quick_start": [],
    "user_guide": [],
    "auto_examples/index": [],
}

html_theme_options = {
    'logo_only': False,
    'display_version': False,
    "external_links": [],
    "github_url": "https://github.com/applied-ml-bde/bde",
    # "twitter_url": ,
    "use_edit_page_button": True,
    "show_toc_level": 1,
    # "navbar_align": "right",  # For testing that the navbar items align properly
}

html_context = {
    "github_user": "applied-ml-bde",
    "github_repo": "bde",
    "github_version": "main",
    "doc_path": "doc",
}

# -- Options for autodoc ------------------------------------------------------

autodoc_typehints = 'description'
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
}

# generate autosummary even if no references
autosummary_generate = True

# -- Options for numpydoc -----------------------------------------------------

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

# -- Options for intersphinx --------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "scikit-learn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

# -- Options for sphinx-gallery -----------------------------------------------

# Generate the plot for the gallery
plot_gallery = True

sphinx_gallery_conf = {
    "doc_module": "bde",
    "backreferences_dir": os.path.join("generated"),
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "reference_url": {"bde": None},
}
