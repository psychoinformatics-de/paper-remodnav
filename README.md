# REMoDNaV: Robust Eye Movement Detection for Natural Viewing

This repository contains the raw data, the code to generate summary statistics, and raw figures for the manuscript, and the manuscript sources for the publication [REMoDNaV: Robust Eye Movement Detection for Natural Viewing](https://doi.org/10.1101/619254).

## Updated instructions for computing the results and building the manuscript

[See below for the original instructions]

More than three years after the journal publication, the original setup for
reproducing the statistical results and figures in the manuscript started to
fail.  Software environments had advanced sufficiently to make changes to the
analysis code necessary. The need to pin the packages used for figure
generation to retain identical outputs further complicated the recreation of a
functional computation environment. To compensate and add longevity, the
computational pipeline was moved to a Docker-based environment.

A protocol of this change is available at
https://github.com/psychoinformatics-de/paper-remodnav/pull/24

The one (minor) difference in the results, compared to the original
publication, is detailed at
https://github.com/psychoinformatics-de/paper-remodnav/issues/20#issuecomment-1757462683

In order to reproduce the results and the manuscript, the following software
needs to be installed:

- DataLad (https://www.datalad.org, verified with v0.19)
- DataLad Containers extension package (https://github.com/datalad/datalad-container, verfied with v1.2.3)
- Docker (https://www.docker.com, verified with v20)

Installation instructions are provided on the respective websites. For DataLad
we recommend the instructions in its handbook at
https://handbook.datalad.org/r?install

This following procedure has been verified to work on Debian, MacOS, and Windows 10.

Obtain this repository with DataLad. It is fully self-contained, and includes
versioned links to all code, data and computational environments:

```
C:\Users\mih>datalad clone https://github.com/psychoinformatics-de/paper-remodnav.git
```

To verify the computational reproducibility of any numerical value reported in the
paper, and the SVG code for all figures, enter the dataset and run:

```
C:\Users\mih>cd paper-remodnav
C:\Users\mih\paper-remodnav>datalad rerun results-containerized
```

This will recompute everything. In order to do this, a total of ~1GB of input data
(detailed in the manuscript) will be downloaded. In addition, about 1.8GB for the
Docker container image are downloaded. Apart from the time needed to download all
information, the actual computation only takes a few minutes.

If recomputation is successful and reproducible, no change to the dataset will be
saved (indicated by `save (notneeded)`). Any bit-precision difference will otherwise
be detected and can be inspected in the form of the change record (last commit).
The to-be-reproduced state is captured by the signed tag `results-containerized`.
https://github.com/psychoinformatics-de/paper-remodnav/releases/tag/results-containerized

The full manuscript can be built with the command:

```
C:\Users\mih\paper-remodnav>datalad containers-run -n docker-make main.pdf
```

## Old instructions for computing the results and building the manuscript

To recompute results and compile the paper, do the following:

- Create a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and activate it:

```
    # one way to create a virtual environment:
    virtualenv --python=python3 ~/env/remodnav
    . ~/env/remodnav/bin/activate
```
 
- ``clone`` the repository with ``git clone https://github.com/psychoinformatics-de/paper-remodnav.git``
- Navigate into the repository and run ``make`` to compile the paper as it was published.
- To recompute results and figures, run ``make clean``, followed by ``make``.

Appropriate Makefiles within the directory will install necessary Python requirements (the ``remodnav`` Python package, ``datalad``, ``pandas``, ``seaborn``, and ``sklearn``), execute data retrieval via [DataLad](http://datalad.org) (about 550MB in total),
compute the results and figures from ``code/mk_figuresnstats.py``, insert the results and rendered figures in the
main.tex file, and render the PDF.
The full PDF will be ``main.pdf``.

### Software requirements

Note that [inkscape](https://inkscape.org/de/release/inkscape-0.92.4/), [latexmk](https://mg.readthedocs.io/latexmk.html),
  and [texlive-latex-extra](https://wiki.ubuntuusers.de/TeX_Live/) need to be installed on your system to render the figures and the PDF.

## Getting help

If you encounter failures, e.g. due to uninstalled python modules, restart ``make`` after running ``make clean``.
If you encounter failures you suspect are due to deficiencies in this repository, please submit an
[issue](https://github.com/psychoinformatics-de/paper-remodnav/issues/new) or a
pull request. Please address issues on bugs or questions of other software to the software's specific home repository.
