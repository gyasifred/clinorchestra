.PHONY: presentation clean

presentation: technical_presentation.pdf

technical_presentation.pdf: technical_presentation.tex
	pdflatex technical_presentation.tex
	pdflatex technical_presentation.tex

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *~

install-deps:
	@echo "Installing LaTeX dependencies..."
	sudo apt-get update
	sudo apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended

help:
	@echo "ClinOrchestra Documentation Build"
	@echo ""
	@echo "Targets:"
	@echo "  presentation  - Compile LaTeX presentation to PDF"
	@echo "  clean         - Remove LaTeX auxiliary files"
	@echo "  install-deps  - Install LaTeX dependencies (requires sudo)"
	@echo ""
	@echo "Usage:"
	@echo "  make install-deps  # First time only"
	@echo "  make presentation  # Compile PDF"
