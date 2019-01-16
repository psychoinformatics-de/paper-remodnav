all: main.pdf

main.pdf: main.tex tools.bib EyeGaze.bib
	latexrun $<

clean:
	rm -f main.pdf
