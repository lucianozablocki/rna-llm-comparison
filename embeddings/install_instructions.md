# Installation instructions
Detalles para uso interno, para poder reproducir los embeddings más adelante. Copiar pesos y código original a (no queda en este repo)

## One-hot
Nothing to install here :)

## ERNIE-RNA

```
git clone https://github.com/Bruce-ywj/ERNIE-RNA.git
cd ./ERNIE-RNA
conda env create -f environment.yml
conda activate ERNIE-RNA
conda install pandas
conda install h5py
python ernie_gen_seq_embedding.py
```

## Rinalmo
