# Installation instructions
Detalles para uso interno, para poder reproducir los embeddings más adelante. Copiar pesos y código original a (no queda en este repo)

## One-hot
Nothing to install here :)

## ERNIE-RNA

```
git clone https://github.com/Bruce-ywj/ERNIE-RNA.git
```

Download pretrained weights from https://drive.google.com/drive/folders/1iX-xtrTtT-zk5je8hCdYQOQHDWbl1wgo and put them under `ERNIE-RNA/checkpoint/ERNIE-RNA_checkpoint`.

```
cd ./ERNIE-RNA
conda env create -f environment.yml
conda activate ERNIE-RNA
conda install pandas
conda install h5py
python ernie-rna.py
```

## RiNALMo
```
git clone https://github.com/lbcb-sci/RiNALMo.git
cd ./RiNALMo
conda create -n "RiNALMo" python=3.11
conda activate RiNALMo
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install .
FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
conda install pandas
conda install h5py
python rinalmo.py
```

## RNA-FM

## RNABERT
##
