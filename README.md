# REMoDNaV: Robust Eye Movement Detection for Natural Viewing

This repository contains the raw data, the code to generate summary statistics, and raw figures for the manuscript, and the manuscript sources for the publication [REMoDNaV: Robust Eye Movement Detection for Natural Viewing](yettolink).

## Manuscript

To recompute results and compile the paper, do the following:

[Optional] create a virtual environment:

    # create and enter a new virtual environment (optional)
    virtualenv --python=python3 ~/env/remodnav
    . ~/env/remodnav/bin/activate
    
- if you haven't yet, install [``remodnav``](https://github.com/psychoinformatics-de/remodnav), ``seaborn``, and
 [``datalad``](https://www.datalad.org). Depending on your operating system, datalad can be installed via
 ``pip install datalad`` or ``sudo apt-get install datalad`` (please check the 
 [docs](http://docs.datalad.org/en/latest/gettingstarted.html) if you are unsure which option is applicable to your system)
 
Install from [PyPi](https://pypi.org/project/remodnav):

    # install from PyPi
    pip install remodnav seaborn sklearn
    # if not installed with another method
    pip install datalad

- ``datalad install`` the repository with ``datalad install https://github.com/psychoinformatics-de/paper-remodnav.git``

- Appropriate Makefiles within the directory will execute data retrieval via datalad (about 550MB in total),
compute the results and figures from ``code/mk_figuresnstats.py``, insert the results and rendered figures in the
main.tex file, and render the PDF with a single call from the root of the directory: ``make``

- Note that [inkscape](https://inkscape.org/de/release/inkscape-0.92.4/), [latexmk](https://mg.readthedocs.io/latexmk.html),
  and [texlive-latex-extra](https://wiki.ubuntuusers.de/TeX_Live/) need to be installed on your system to render the figures and the     PDF.

The full PDF will be ``main.pdf``.
 

If you encounter failures, e.g. due to uninstalled python modules, restart ``make`` after running ``make clean``.
If you encounter failures you suspect are due to deficiencies in this repository, please submit an
[issue](https://github.com/psychoinformatics-de/paper-remodnav/issues/new) or a
pull request. Please address issues on bugs or questions of other software to the software's specific home repository.
