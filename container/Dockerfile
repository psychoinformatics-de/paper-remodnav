FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq -y --allow-releaseinfo-change && \
# basic tooling we need for building/installing
apt-get install -q --no-install-recommends -y make && \
# for compiling custom python package installs
apt-get install -q --no-install-recommends -y build-essential python3-dev && \
# any python package we do NOT need to compile
apt-get install -q --no-install-recommends -y python3-virtualenv python3-wheel python3-scipy python3-sklearn python3-statsmodels python3-pyparsing python3-pil python3-pip && \
# now python packages that we need specific versions of
# we need the following packages pinned
# - matplotlib for a particular SVG file content (structure)
# - pandas as the last per v2 release for an API incompatibility that with 2x
# - seaborn for minor changes in figure composition in later versions
# - and numpy...the version does not matter, but it has to be different from
#   the Debian-provided one, such that pip will compile it
#   full and long story at
#   https://github.com/psychoinformatics-de/paper-remodnav/issues/20
python3 -m pip install --break-system-packages numpy==1.24.3 pandas==1.5.3 seaborn==0.10.1 matplotlib==3.4.3 && \
# now the pieces we need to build the PDF
apt-get install -q --no-install-recommends -y inkscape latexmk texlive-latex-extra && \
# trim the image size
rm -rf /root/.local /root/.cache /var/lib/apt/lists/deb.debian.org* ; \
apt-get purge -y build-essential ; \
apt-get autoremove -y ; \
apt-get clean
