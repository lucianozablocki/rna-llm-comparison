# rna-llm-comparison
repository that holds the code to build a secondary structure RNA predictor model based on LLM representations.

# Train the model

## Prerequisites

With an Anaconda working installation, run:

```
conda env create -f environment.yml
```

This should install all required dependencies.

You will also need to download representations, check the links in [Embeddings section](README.md#embeddings)

## Generate train and test partitions

We will use a family fold approach: for each one of the 9 families under consideration, one will be left out for test. This way, we have 9 train and test partitions. To generate them, run:

```
python gen_splits.py
```

This will create the partitions under `data/archiveII_famfold`.

## Train the model

The `train_model.py` script contains a few command line parameters, an example to run the train for 15 epochs using RiNALMo-generated representations, with tmRNA family as held out test partition set is:

```
python train_model.py --train_partition_path data/archiveII_famfold/tmRNA/train.csv --test_partition_path data/archiveII_famfold/tmRNA/test.csv --out_path <directory_where_to_write_results>
```

## Embeddings
ERNIE-RNA (TODO: hacer carpetas publicas para esto)
https://drive.google.com/file/d/1eKL5hsc0vXDr4GUGV2dlAb6t6PhQ5qD1/view?usp=sharing
RINALMO
https://drive.google.com/file/d/1lBZDnPPcGMymkck6amwCWTDMYqD8EhHA/view?usp=sharing
