#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = or-p1
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# PATHS                                                                     #
#################################################################################

DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/00_raw
RAW_DATA_ZIP_FILE = $(DATA_DIR)/VOCtrainval_11-May-2012.tar
STAMPS_DIR = $(DATA_DIR)/stamps

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## download, unzip and arrange the raw data
.PHONY: setup_raw_data
setup_raw_data: clear_raw_data ${STAMPS_DIR}/raw_data_unpacked

## remove all raw data
.PHONY: clear_raw_data
clear_raw_data:
	rm -rf $(RAW_DATA_DIR)
	rm -f $(STAMPS_DIR)/raw_data_unpacked $(STAMPS_DIR)/raw_data_dir

## remove all stamps
.PHONY: clear_stamps
clear_stamps:
	rm -rf $(STAMPS_DIR)

#################################################################################
# DATA DEPENDENCIES                                                             #
#################################################################################

${STAMPS_DIR}/stamps_dir:
	mkdir -p $(STAMPS_DIR)
	touch $@

${STAMPS_DIR}/raw_data_dir: ${STAMPS_DIR}/stamps_dir
	mkdir -p $(RAW_DATA_DIR)
	touch $@

${STAMPS_DIR}/raw_data_zip_file: ${STAMPS_DIR}/raw_data_dir
	@echo "Checking if $(RAW_DATA_ZIP_FILE) exists..."
	if [ ! -f $(RAW_DATA_ZIP_FILE) ]; then \
		echo "File does not exist. Downloading..."; \
		curl -L http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -o $(RAW_DATA_ZIP_FILE); \
	else \
		echo "File already exists. Skipping download."; \
	fi
	touch $@

${STAMPS_DIR}/raw_data_unpacked: ${STAMPS_DIR}/raw_data_zip_file
	tar -xvf $(RAW_DATA_ZIP_FILE) -C $(RAW_DATA_DIR)
	mv $(RAW_DATA_DIR)/VOCdevkit/VOC2012/* $(RAW_DATA_DIR)/
	rm -rf $(RAW_DATA_DIR)/VOCdevkit
	touch $@

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