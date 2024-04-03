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


## Run Training
This pipeline starts training a semantic-segmentation model.  
Training-parameters can be seen or, modified from `training_config.yaml`.

#### Check help
`python3 melseg_trainer.py --help`

#### Command example
`python3 melseg_trainer.py --config training_config.yaml`

`python3 melseg_trainer.py --config example_configs/training_config_01.yaml --cuda_id 0`


## Run Analysis
This pipeline analyzes the *training-logs [.csv]* of a cross-validation run, by:
- finding the best performance of each fold using **minimum loss** and **maximum epoch**
- and, calculating average of fold-wise best performances.
This pipeline saves the analysis report as a *[.csv]* file.

#### Check help
`python3 melseg_analyzer.py --help`

#### Command example
`python3 melseg_analyzer.py --dir example_logs/ --out example_logs/log_analysis.csv`

or,

`python3 melseg_analyzer.py --dir example_logs/`


## Run Inference
This pipeline runs inference using trained melSeg models.

#### Check help
`python3 melseg_inference.py --help`

#### Command example
`python3 melseg_inference.py --checkpoint ../checkpoint_fold3.pt --image_dir ../datasets/HAM50/images/ --mask_dir ../datasets/HAM50/masks/ --out_dir ../datasets/HAM50/results/`


## Split Config
This pipeline generates individual sweeps from a training-config.

#### Check help
`python3 config_splitter.py --help`

#### Command example
`python3 config_splitter.py --config training_config.yaml --outdir example_configs`

## Code Formatting
**Black** is a Python code formatter that enforces a consistent style by rewriting code to conform to **PEP 8** style guide.

#### Command
`sh run_black.sh`

## Code Linting
Linting is the process of static code analysis that examines code for errors, stylistic issues, and potential problems. There are many different linters available for Python, **Flake8** is used here.

#### Command
`sh run_flake8.sh`

## Metric Calculation
Following metrics are calculated while training the model.   

#### PRECISION
Precision score is the number of true positive results divided by the number of all positive results.

$$Precision=TP/(TP+FP)$$

#### RECALL
Recall score, also known as Sensitivity or true positive rate, is the number of true positive results 
divided by the number of all samples that should have been identified as positive.

$$Recall=TP/(TP+FN)$$

#### ACCURACY
Accuracy score, also known as Rand index is the number of correct predictions, consisting of correct 
positive and negative predictions divided by the total number of predictions.

$$Accuracy=(TP+TN)/(TP+TN+FP+FN)$$

#### DICE COEFFICIENT (F1-SCORE)
Dice coefficient, also known as F1 score is the harmonic mean of precision and recall. 
In other words, it is calculated by *2 x intersection* divided by the *total number of pixel in both images*.
$$Dice=2TP/(2TP+FP+FN)$$

#### INTERSECTION OVER UNION (IoU)
Intersection over Union (IoU), also known as Jaccard index is the area of the intersection over union of the predicted segmentation and the ground truth.
$$IoU=TP/(TP+FP+FN)$$

*Ref: https://www.kaggle.com/code/nghihuynh/understanding-evaluation-metrics-in-segmentation/notebook*
