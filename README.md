# REMoDNaV: Robust Eye Movement Detection for Natural Viewing

This repository contains the raw data, the code to generate summary statistics, and raw figures for the manuscript, and the manuscript sources for the publication [REMoDNaV: Robust Eye Movement Detection for Natural Viewing](yettolink).

## Manuscript

To recompute results and compile the paper, do the following:

- Create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and activate it:

```
    # one way to create a virtual environment:
    virtualenv --python=python3 ~/env/remodnav
    . ~/env/remodnav/bin/activate
```
 
- ``clone`` the repository with ``git clone https://github.com/psychoinformatics-de/paper-remodnav.git``
- Navigate into the repository and run ``make``

Appropriate Makefiles within the directory will install necessary Python requirements (the ``remodnav`` Python package, ``datalad``, ``pandas``, ``seaborn``, and ``sklearn``), execute data retrieval via [DataLad](http://datalad.org) (about 550MB in total),
compute the results and figures from ``code/mk_figuresnstats.py``, insert the results and rendered figures in the
main.tex file, and render the PDF.
The full PDF will be ``main.pdf``.

## Software requirements

Note that [inkscape](https://inkscape.org/de/release/inkscape-0.92.4/), [latexmk](https://mg.readthedocs.io/latexmk.html),
  and [texlive-latex-extra](https://wiki.ubuntuusers.de/TeX_Live/) need to be installed on your system to render the figures and the PDF.

## Getting help

If you encounter failures, e.g. due to uninstalled python modules, restart ``make`` after running ``make clean``.
If you encounter failures you suspect are due to deficiencies in this repository, please submit an
[issue](https://github.com/psychoinformatics-de/paper-remodnav/issues/new) or a
pull request. Please address issues on bugs or questions of other software to the software's specific home repository.
