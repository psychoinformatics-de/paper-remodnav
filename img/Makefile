all: pics

# use `chronic` to make output look neater, if available
CHRONIC=$(shell which chronic || echo '' )

# For every .svg we must have a pdf
pics: $(shell find . -iname \*.svg | sed -e 's/svg/pdf/g' )

clean:
	for p in *.svg; do rm -f $${p%*.svg}.eps $${p%*.svg}.pdf; done
	-rm -rf *_300dpi.png

# git-ignore each auto-rendered figure
.PHONY: ignore-%
ignore-%:
	@grep -q "^$*$$" .gitignore || { \
	  echo "$*" >> .gitignore; echo "Ignore $@"; }

#
# Inkscape rendered figures
#
# try modern-age API first, fall-back on old one, if needed
%.pdf: %.svg ignore-%.pdf
	@echo "Rendering $@"
	@$(CHRONIC) inkscape --export-filename="$@" "$<" || $(CHRONIC) inkscape -z -f "$<" -A "$@"

.PHONY: all pics
