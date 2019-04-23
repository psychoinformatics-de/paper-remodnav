all: main.pdf

main.pdf: main.tex tools.bib EyeGaze.bib results_def.tex figures
	latexmk -pdf -g $<

results_def.tex: code/anderson.py
	bash -c 'set -o pipefail; code/anderson.py -f -r -s -m | tee results_def.tex'

figures: code/anderson.py
	$(MAKE) -C img

clean:
	rm -f main.bbl main.aux main.blg main.log main.out main.pdf main.tdo main.fls main.fdb_latexmk example.eps img/*eps-converted-to.pdf texput.log results_def.tex
	$(MAKE) -C img clean


.PHONY: figures
