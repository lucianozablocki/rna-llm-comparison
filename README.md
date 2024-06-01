# rna-llm-comparison
repository that holds the code to build a secondary structure RNA predictor model based on LLM representations.

# Train the model

## Prerequisites

With an Anaconda working installation, run:

```
conda env create -f environment.yml
```

This should install all required dependencies.

You will also need to download representations from `data` (`all_repr_ERNIE-RNA.txt` and `all_repr_archiveii_RiNALMo.txt`).

## Generate train and test partitions

We will use a family fold approach: for each one of the 9 families under consideration, one will be left out for test. This way, we have 9 train and test partitions. To generate them, run:

```
python gen_splits.py
```

This will create the partitions under `data/archiveII_famfold`.

## Train the model

The `ss-model.py` script contains a few command line parameters, an example to run the train for 15 epochs using RiNALMo-generated representations, with tmRNA family as held out test partition set is:

```
python ss-model.py --train_partition_path data/archiveII_famfold/tmRNA/train.csv --out_path <directory_where_to_write_results>
```
