# docs/source/conf.py

# 项目信息
project = 'pointgluecpp'
project_copyright = '2026, ssl'
author = 'ssl'
release = '1.0.0'
version = '1.0.0'

# 扩展配置
extensions = [
    'breathe'
]

# Breathe配置（Doxygen集成）
breathe_projects = {
    "pointgluecpp": "../build/xml"
}
breathe_default_project = "pointgluecpp"

language = 'zh_CN'

html_theme = 'sphinx_rtd_theme'
