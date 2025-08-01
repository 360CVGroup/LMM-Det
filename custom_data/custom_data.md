# Data Curation

In Stage IV, we curate a customized dataset to make LMM-Det excel in object detection while preserving its inherent capabilities like caption generation and VQA.

## Step 1

We generate pesudo labels on the trainset of COCO using [Salience-DETR](https://github.com/xiuqhou/Salience-DETR) (FocalNet-L backone), and re-organize them into a instruction format. Note that the re-organization data consists of ground-truth labels and pesudo labels.
(In practice, this data is aslo used in Stage III.)

## Step 2

We remove the textcaps data in the LLaVA-665K instruction data.

## Step 3

We concat the the re-organization data and the LLaVA-665K instruction data (without textcaps) as the training data in Stage IV.