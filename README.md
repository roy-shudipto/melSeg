# melSeg
This is a tool to train semantic-segmentation model targeting skin leison segmentation in digital images.

## Install requirements
`pip install -r requirements.txt`

## Download and structure HAM10000 Dataset
- Download Link: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 
- Unzip all zipped directories from the download.
- Create a directory called **HAM10000** with two sub-directories: **images, masks**
- Move all RGB-images to the sub-directory: **HAM10000/images**
- Move all Mask-images to the sub-directory: : **HAM10000/masks**


## Run training
This pipeline will start training a semantic-segmentation model.  
Training-parameters can be seen or, modified from `training_config.yaml`.

### Check help
`python3 melseg_trainer.py --help`

### Command example
`python3 melseg_trainer.py --config training_config.yaml`
