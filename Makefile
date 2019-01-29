all: main.pdf

main.pdf: main.tex tools.bib EyeGaze.bib
	latexmk -pdf -g $<

clean:
	rm -f main.pdf
