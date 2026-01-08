# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import subprocess

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PointGlueCpp'
copyright = '2026, PointGlueCpp Contributors'
author = 'PointGlueCpp Contributors'
release = '1.0.0'
version = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'breathe',
    'exhale',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Breathe 配置 (Doxygen -> Sphinx) ------------------------------------

# 运行 Doxygen
def run_doxygen():
    """运行 Doxygen 生成 XML"""
    doxygen_file = '../Doxyfile'
    if os.path.exists(doxygen_file):
        try:
            subprocess.call(['doxygen', doxygen_file])
        except FileNotFoundError:
            print("Warning: doxygen not found. Please install doxygen.")
    else:
        print(f"Warning: {doxygen_file} not found")

# 在构建前运行 Doxygen
def setup(app):
    app.connect('builder-inited', run_doxygen)

# Breathe 配置
breathe_projects = {
    'PointGlueCpp': '../xml'
}

breathe_default_project = 'PointGlueCpp'

# 指定 Doxygen 项目路径
breathe_domain_by_extension = {
    'h': 'cpp',
    'cc': 'cpp',
    'cpp': 'cpp',
}

# -- Exhale 配置 (生成漂亮的 API 文档) -----------------------------------

exhale_args = {
    # 这些核心参数是必须的
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxyfileEncoding": "UTF-8",
    "createTreeView": True,
    # Exhale 告诉 Breathe 这些是 Doxygen 项目
    "breatheProjects": breathe_projects,
    # HTML 生成设置
    "treeViewIsBootstrap": True,
    # LaTeX 设置
    "contentsTitle": "Page Contents",
    "katexStyle": "github.com/Khan/KaTeX/0.9.0/katex.min.css",
    "katexScript": "github.com/Khan/KaTeX/0.9.0/katex.min.js",
}

# -- Napoleon 配置 (支持 Google/NumPy 风格注释) ---------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx 配置 (链接到其他项目文档) -------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'opencv': ('https://docs.opencv.org/4.x/', None),
}

# -- 其他设置 -------------------------------------------------------------

# 语言设置
language = 'zh_CN'

# 添加源文件后缀
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# 主文档
master_doc = 'index'

# Pygments (语法高亮) 主题
pygments_style = 'sphinx'

# 保持文档顺序
todo_include_todos = True