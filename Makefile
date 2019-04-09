all: main.pdf

main.pdf: main.tex tools.bib EyeGaze.bib results_def.tex
	latexmk -pdf -g $<

results_def.tex: code/anderson.py
	code/anderson.py -f -r -s \
        | tee results_def.tex

clean:
	rm -f main.bbl main.aux main.blg main.log main.out main.pdf main.tdo main.fls main.fdb_latexmk example.eps img/*eps-converted-to.pdf texput.log
