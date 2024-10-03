# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import toml
import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[1].resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dtaianomaly'
copyright = '2023, DTAI'
author = 'Louis Carpentier \\and Nick Seeuws'

with open('../pyproject.toml', 'r') as f:
    config = toml.load(f)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'numpydoc'
]

numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin theme
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'dtaianomalydoc'

# -- Versioning control of documentation ------------------------------------------
# https://www.codingwiththomas.com/blog/my-sphinx-best-practice-for-a-multiversion-documentation-in-different-languages


# Load all the versions to build the documentation from
with open('versions.toml', 'r') as versions_toml:
    raw_versions = sorted(toml.load(versions_toml)['versions'], reverse=True)

# Add link to the version
current_version = os.environ.get("current_version", "latest")
if current_version == 'latest':
    versions = [('latest', '.')] + [(version, version.replace('.', '_')) for version in raw_versions]
else:
    versions = [('latest', '../')] + [
        (version, ('../' + version.replace('.', '_')) if version != current_version else '.')
        for version in raw_versions
    ]

html_context = {
    'current_version': current_version,
    'versions': versions
}
version = current_version
