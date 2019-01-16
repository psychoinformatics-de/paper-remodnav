all: main.pdf

main.pdf: main.tex
	latexrun $<

clean:
	rm -f main.pdf
