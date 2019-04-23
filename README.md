# paper-remodnav

This repository contains all analysis code for results and figures presented in the publication
[REMoDNaV: Robust Eye Movement Detection for Natural Viewing](yettolink),
and the LaTeX files necessary to compile the full PDF. 
To obtain the input data used for the analysis, recompute the results, and recompile the PDF with
all results and figures, do the following:

[Optional] create a virtual environment:

    # create and enter a new virtual environment (optional)
    virtualenv --python=python3 ~/env/remodnav
    . ~/env/remodnav/bin/activate
    
- if you haven't yet, install [``remodnav``](https://github.com/psychoinformatics-de/remodnav)
  and its dependecies (numpy, matplotlib, statsmodels, scipy, pandas):
 
 
Install from [PyPi](https://pypi.org/project/remodnav) with all dependencies:

    # install from PyPi
    pip install remodnav


- Additionally, please install ``seaborn`` (``pip install seaborn``) and
 [``datalad``](https://www.datalad.org). Depending on your operating system, datalad can be installed via
  ``pip install datalad`` or ``sudo apt-get install datalad`` (please check the [docs](http://docs.datalad.org/en/latest/gettingstarted.html)
  if you are unsure which option is applicable to your system)
- datalad install this repository and its subdatasets:

Installing recursively with

     # with SSH
     datalad install -r git@github.com:psychoinformatics-de/paper-remodnav.git
     # or using HTTPS
     datalad install -r https://github.com/psychoinformatics-de/paper-remodnav.git


- Note: To render figures, [``inkscape``](https://inkscape.org/de/) has to be installed on your system;
  To compile the PDF, ``pdflatex`` has to be installed.

Appropriate Makefiles within the directory will execute data retrieval via datalad (about 550MB in total),
compute the results and figures from ``code/mk_figuresnstats.py``, insert the results and rendered figures in the
main.tex file, and render the PDF with a single call:

In the root of the directory, run

    make -B
    
 

If you encounter failures, e.g. due to uninstalled python modules, restart ``make`` after running ``make clean``.
If you encounter failures you suspect are due to deficiencies in this repository, please submit an
[issue](https://github.com/psychoinformatics-de/paper-remodnav/issues/new) or a
pull request. Please address issues on bugs or questions of other software to the software's specific home repository.
