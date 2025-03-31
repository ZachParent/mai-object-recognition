#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = p2
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
SUBMISSION_NAME = AgundezLangParentRecaldeSanchez_OR_P2
ZIP_NAME = $(SUBMISSION_NAME).zip

#################################################################################
# PATHS                                                                     #
#################################################################################

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Create distribution zip
.PHONY: zip
zip:
	rm -f $(ZIP_NAME) 
	cd .. && \
	mkdir -p $(SUBMISSION_NAME) && \
	cp -r $(PROJECT_NAME)/* $(SUBMISSION_NAME)/ && \
	zip -r $(SUBMISSION_NAME).zip $(SUBMISSION_NAME) \
			-x "*.aux" "*.log" "*.out" "*.toc" "*.gz" "*.tar" \
			"*.fls" "*.fdb_latexmk" "*.synctex.gz" \
			"*/__pycache__/*" "*.pyc" \
			"*.egg-info/*" "*/TODO.md" \
			"*venv/*" "*/.git/*" \
			"*/data/00_raw/*" "*/data/01_runs/*" \
			"*/data/04_models/*" && \
	rm -rf $(SUBMISSION_NAME) && \
	mv $(SUBMISSION_NAME).zip ${PROJECT_NAME}/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)