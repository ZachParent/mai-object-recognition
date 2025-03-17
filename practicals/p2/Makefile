#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = p2
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python
SUBMISSION_NAME = AgundezLangParentRecaldeSanchez_OR_P2
ZIP_NAME = $(SUBMISSION_NAME).zip

#################################################################################
# PATHS                                                                         #
#################################################################################

DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/00_raw

# Image and annotation directories following the standard Fashionpedia structure
IMG_DIR = $(RAW_DATA_DIR)/images
IMG_TRAIN_DIR = $(IMG_DIR)/train
IMG_VAL_DIR = $(IMG_DIR)/val
ANNOT_DIR = $(RAW_DATA_DIR)/annotations

STAMPS_DIR = $(DATA_DIR)/stamps

#################################################################################
# DATA FILES                                                                    #
#################################################################################

TRAIN_ZIP_URL       = https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
VALTEST_ZIP_URL     = https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
TRAIN_ANNOT_URL     = https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
VALTEST_ANNOT_URL   = https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json

TRAIN_ZIP       = $(IMG_DIR)/train2020.zip
VALTEST_ZIP     = $(IMG_DIR)/val_test2020.zip
TRAIN_ANNOT     = $(ANNOT_DIR)/instances_attributes_train2020.json
VALTEST_ANNOT   = $(ANNOT_DIR)/instances_attributes_val2020.json

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Download, extract and arrange raw data
.PHONY: setup_raw_data
setup_raw_data: clear_raw_data \
    $(STAMPS_DIR)/train_zip_downloaded \
    $(STAMPS_DIR)/valtest_zip_downloaded \
    $(STAMPS_DIR)/train_annot_downloaded \
    $(STAMPS_DIR)/valtest_annot_downloaded \
    $(STAMPS_DIR)/images_unpacked

## Format code using black
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m black src

## Remove all raw data and stamp files, then recreate necessary directories
.PHONY: clear_raw_data
clear_raw_data:
	rm -rf $(RAW_DATA_DIR)
	rm -rf $(STAMPS_DIR)
	mkdir -p $(IMG_TRAIN_DIR)
	mkdir -p $(IMG_VAL_DIR)
	mkdir -p $(ANNOT_DIR)
	mkdir -p $(STAMPS_DIR)
	rm -f $(STAMPS_DIR)/train_zip_downloaded \
          $(STAMPS_DIR)/valtest_zip_downloaded \
          $(STAMPS_DIR)/train_annot_downloaded \
          $(STAMPS_DIR)/valtest_annot_downloaded \
          $(STAMPS_DIR)/images_unpacked

## Download train images zip
$(STAMPS_DIR)/train_zip_downloaded: $(IMG_DIR)
	@echo "Downloading train2020.zip..."
	if [ ! -f $(TRAIN_ZIP) ]; then \
		curl -L $(TRAIN_ZIP_URL) -o $(TRAIN_ZIP); \
	else \
		echo "train2020.zip already exists. Skipping download."; \
	fi
	touch $@

## Download val/test images zip
$(STAMPS_DIR)/valtest_zip_downloaded: $(IMG_DIR)
	@echo "Downloading val_test2020.zip..."
	if [ ! -f $(VALTEST_ZIP) ]; then \
		curl -L $(VALTEST_ZIP_URL) -o $(VALTEST_ZIP); \
	else \
		echo "val_test2020.zip already exists. Skipping download."; \
	fi
	touch $@

## Download train annotations JSON
$(STAMPS_DIR)/train_annot_downloaded: $(ANNOT_DIR)
	@echo "Downloading instances_attributes_train2020.json..."
	if [ ! -f $(TRAIN_ANNOT) ]; then \
		curl -L $(TRAIN_ANNOT_URL) -o $(TRAIN_ANNOT); \
	else \
		echo "instances_attributes_train2020.json already exists. Skipping download."; \
	fi
	touch $@

## Download val/test annotations JSON
$(STAMPS_DIR)/valtest_annot_downloaded: $(ANNOT_DIR)
	@echo "Downloading instances_attributes_val2020.json..."
	if [ ! -f $(VALTEST_ANNOT) ]; then \
		curl -L $(VALTEST_ANNOT_URL) -o $(VALTEST_ANNOT); \
	else \
		echo "instances_attributes_val2020.json already exists. Skipping download."; \
	fi
	touch $@

## Unpack image ZIP files into the appropriate folders using rsync to avoid wildcard issues
$(STAMPS_DIR)/images_unpacked: $(STAMPS_DIR)/train_zip_downloaded $(STAMPS_DIR)/valtest_zip_downloaded
	@echo "Unzipping train2020.zip..."
	unzip -q $(TRAIN_ZIP) -d $(IMG_DIR)/temp_train
	# If the zip has a subfolder named 'train/', rsync its contents to IMG_TRAIN_DIR
	if [ -d "$(IMG_DIR)/temp_train/train" ]; then \
		rsync -a $(IMG_DIR)/temp_train/train/ $(IMG_TRAIN_DIR)/; \
	else \
		@echo "No 'train/' subfolder found; adjust as needed."; \
	fi
	rm -rf $(IMG_DIR)/temp_train

	@echo "Unzipping val_test2020.zip..."
	unzip -q $(VALTEST_ZIP) -d $(IMG_DIR)/temp_val
	# For val_test2020.zip, since it contains a 'test/' folder, move its contents to IMG_VAL_DIR
	if [ -d "$(IMG_DIR)/temp_val/test" ]; then \
		rsync -a $(IMG_DIR)/temp_val/test/ $(IMG_VAL_DIR)/; \
	else \
		@echo "No 'test/' subfolder found; adjust as needed."; \
	fi
	rm -rf $(IMG_DIR)/temp_val

	touch $@

#################################################################################
# Create distribution zip                                                       #
#################################################################################

.PHONY: zip
zip:
	rm -f $(ZIP_NAME)
	cd .. && \
	mkdir -p $(SUBMISSION_NAME) && \
	cp -r $(PROJECT_NAME)/* $(SUBMISSION_NAME)/ && \
	cp ../requirements.txt $(SUBMISSION_NAME)/ && \
	cp ../requirements_cuda.txt $(SUBMISSION_NAME)/ && \
	zip -r $(SUBMISSION_NAME).zip $(SUBMISSION_NAME) \
		-x ".aux" ".log" ".out" ".toc" ".gz" ".tar" \
		".fls" ".fdb_latexmk" ".synctex.gz" \
		"/pycache/" ".pyc" \
		".egg-info/" "/TODO.md" \
		"/.venv/" "/.git/" \
		"/data/00_raw/" "/data/stamps/*" && \
	rm -rf $(SUBMISSION_NAME) && \
	mv $(SUBMISSION_NAME).zip $(PROJECT_NAME)/

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELPPYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
