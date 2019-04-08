all: main.pdf paper-data

main.pdf: main.tex tools.bib EyeGaze.bib
	latexmk -pdf -g $<

paper-data:
	code/anderson.py -f True -r True -s True \
        | tee results_def.tex

clean:
	rm -f main.bbl main.aux main.blg main.log main.out main.pdf main.tdo main.fls main.fdb_latexmk example.eps img/*eps-converted-to.pdf texput.log
