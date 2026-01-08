# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# 生成 HTML 文档
html:
	@echo "Generating HTML documentation..."
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "HTML documentation generated in $(BUILDDIR)/html"
	@echo "Open $(BUILDDIR)/html/index.html in your browser"

# 生成 PDF 文档
pdf:
	@echo "Generating PDF documentation..."
	$(SPHINXBUILD) -b latex "$(SOURCEDIR)" "$(BUILDDIR)/latex" $(SPHINXOPTS)
	@echo "Running LaTeX to generate PDF..."
	$(MAKE) -C "$(BUILDDIR)/latex" all-pdf
	@echo "PDF documentation generated in $(BUILDDIR)/latex"

# 清理构建文件
clean:
	@echo "Cleaning build files..."
	rm -rf $(BUILDDIR)/*
	@echo "Clean complete"

# 实时预览（需要安装 sphinx-autobuild）
livehtml:
	@echo "Starting live preview server..."
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

# 检查链接
linkcheck:
	@echo "Checking external links..."
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)
	@echo "Link check complete"

.PHONY: help html pdf clean livehtml linkcheck