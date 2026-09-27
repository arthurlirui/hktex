# Makefile for the Spline-Texture draft.
# Build:  make
# Clean:  make clean
#
# Requires TeX Live (pdflatex, bibtex) and the acmart class on TEXINPUTS.
ACMART  := ../acmart
ROOT    := ..
TEXINPUTS := $(ACMART)//;$(ROOT)//;
BSTINPUTS := $(ROOT)//;

MAIN := main

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex related_work.tex ../references.bib
	TEXINPUTS="$(TEXINPUTS)" pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	BSTINPUTS="$(BSTINPUTS)" bibtex $(MAIN)
	TEXINPUTS="$(TEXINPUTS)" pdflatex -interaction=nonstopmode $(MAIN).tex
	TEXINPUTS="$(TEXINPUTS)" pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).toc
