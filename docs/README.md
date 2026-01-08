# PointGlueCpp 文档

这个目录包含 PointGlueCpp 项目的 Sphinx + Doxygen 文档。

## 本地构建文档

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 安装 Python 依赖
pip install -r requirements.txt

# 确保系统安装了 Doxygen
# Ubuntu/Debian:
sudo apt install doxygen graphviz
# macOS:
brew install doxygen graphviz
# Windows: 从 https://www.doxygen.nl/ 下载安装
```

### 2. 构建 HTML 文档

```bash
cd docs
make html
```

生成的文档在 `_build/html/index.html`。

### 3. 构建 PDF 文档

```bash
make pdf
```

生成的 PDF 在 `_build/latex/` 目录。

### 4. 实时预览（开发时使用）

```bash
make livehtml
```

打开浏览器访问 http://127.0.0.1:8000

### 5. 清理构建文件

```bash
make clean
```

## 部署到 Read the Docs

1. 将项目推送到 GitHub
2. 在 https://readthedocs.org/ 注册账号
3. 导入项目
4. Read the Docs 会自动使用 `.readthedocs.yaml` 配置构建文档

## 文档结构

```
docs/
├── conf.py              # Sphinx 配置
├── index.rst            # 首页
├── overview.rst         # 概述
├── installation.rst     # 安装指南
├── examples.rst         # 示例代码
├── Makefile             # 构建脚本
├── requirements.txt     # Python 依赖
├── _build/              # 构建输出（gitignore）
├── _static/             # 静态文件
├── _templates/          # 模板文件
└── api/                 # API 文档（由 Exhale 生成）
```

## 添加新文档

1. 在 `docs/` 目录创建 `.rst` 文件
2. 在 `index.rst` 的 `toctree` 中添加引用

## 更新 API 文档

API 文档由 Doxygen 和 Breathe 自动生成：

1. 修改 C++ 代码中的 Doxygen 注释
2. 运行 `make html`
3. API 文档会自动更新

## 主题定制

项目使用 Read the Docs 主题。要自定义样式：

1. 在 `_static/` 创建自定义 CSS
2. 在 `conf.py` 中设置 `html_css_files`

```python
html_css_files = [
    'custom.css',
]
```