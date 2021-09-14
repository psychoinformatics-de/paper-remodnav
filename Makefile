# use `chronic` to make output look neater, if available
CHRONIC=$(shell which chronic || echo '' )

PYTHON=python


all: main.pdf

# important to process stats and figures first, such that
# up-to-date versions are compiled into the manuscript
main.pdf: main.tex results_def.tex references.bib 
	@echo "# Render figures"
	$(MAKE) -C img
	@echo "# Render manuscript"
	@$(CHRONIC) latexmk -pdf -g $<

# the stats-script outputs all scores and figures
results_def.tex: code/mk_figuresnstats.py
	@test -z "$$VIRTUAL_ENV" && \
		echo "ERROR: must be executed in a virtual env (set VIRTUAL_ENV to fake one)" && \
		exit 1 || true
	@echo "# Ensure REMODNAV installation"
	@python -m pip install pandas==1.0.5 seaborn==0.10.1 scikit-learn==0.23.0 datalad
	@datalad get -n remodnav
	@$(CHRONIC) pip install -e remodnav
	@rm -f $@
	@REMODNAV_RESULTS=$@ $(PYTHON) code/mk_figuresnstats.py -s -f -r -m

clean:
	rm -f main.bbl main.aux main.blg main.log main.out main.pdf main.tdo \
		main.fls main.fdb_latexmk texput.log \
		results_def.tex
	$(MAKE) -C img clean

virtualenv:

.PHONY: clean
