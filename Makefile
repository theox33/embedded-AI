# Makefile pour générer le PDF du rapport à partir de rapport.md

RAPPORT=rapport.md
PDF=rapport.pdf

BLOCK_DIAGRAM=block_diagram.plantuml
BLOCK_IMG=block_diagram.png

$(BLOCK_IMG): $(BLOCK_DIAGRAM)
	plantuml -tpng $(BLOCK_DIAGRAM)

all: $(PDF)

image: $(BLOCK_IMG)

$(PDF): $(RAPPORT)
	pandoc $< -o $@ --pdf-engine=xelatex --metadata-file=pandoc-marges.yaml -H pandoc-style.tex

clean:
	rm -f $(PDF)

.PHONY: all clean
