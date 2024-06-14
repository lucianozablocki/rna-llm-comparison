# rna-llm-comparison
Code to build a secondary structure RNA predictor model based on LLM representations.

## Prerequisites

```
git clone https://github.com/lucianozablocki/rna-llm-comparison
cd rna-llm-comparison/
```

With an Anaconda working installation, run:

```
conda env create -f environment.yml
```

This should install all required dependencies.

```
conda activate rna-llm
```

You will also need to download representations, check the links in [Embeddings section](README.md#embeddings)

## Generate train and test partitions for ARchiveII

We will use a family fold approach: for each one of the 9 families under consideration, one will be left out for test. This way, we have 9 train and test partitions. To generate them, run:

```
python gen_splits.py
```

This will create the partitions under `data/archiveII_famfold`.

## Train/test scripts

Save the embeddings in the data directory (for example `data/rnafm_ArchiveII.h5 format`).
Then edit the `scripts/run_famfold.py` to specify the embedding name (for example `emb_name = "rnafm"`).

ArchiveII fam-fold train and test:

```
python scripts/run_famfold.py
```

bpRNA TR0/TS0 train and test:

```
python scripts/run_bprna.py
```

## Only train the model

The `train_model.py` script contains a few command line parameters, an example to run the train for 15 epochs using RiNALMo-generated representations, with tmRNA family as held out test partition set is:

```
python train_model.py --train_partition_path data/archiveII_famfold/tmRNA/train.csv --test_partition_path data/archiveII_famfold/tmRNA/test.csv --out_path <directory_where_to_write_results>
```

## Embeddings
ERNIE-RNA (TODO: hacer carpetas publicas para esto)
https://drive.google.com/file/d/1eKL5hsc0vXDr4GUGV2dlAb6t6PhQ5qD1/view?usp=sharing
RINALMO
https://drive.google.com/file/d/1lBZDnPPcGMymkck6amwCWTDMYqD8EhHA/view?usp=sharing


