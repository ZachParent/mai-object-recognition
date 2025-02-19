## VARIABLES

DATA_DIR = data
RAW_DATA_DIR = $(DATA_DIR)/00_raw
RAW_DATA_ZIP_FILE = $(RAW_DATA_DIR)/VOCtrainval_11-May-2012.tar

## COMMANDS

${RAW_DATA_ZIP_FILE}:
	mkdir -p $(RAW_DATA_DIR)
	curl -L http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -o $(RAW_DATA_ZIP_FILE)

setup_raw_data: ${RAW_DATA_ZIP_FILE}
	tar -xvf $(RAW_DATA_ZIP_FILE) -C $(RAW_DATA_DIR)
	mv $(RAW_DATA_DIR)/VOCdevkit/VOC2012/* $(RAW_DATA_DIR)/
	rm -rf $(RAW_DATA_DIR)/VOCdevkit
