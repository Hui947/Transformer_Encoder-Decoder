# Transformer_Encoder-Decoder
Building the transformer architecture from scratch

## Requirements
### Installation
```bash
pip install -r requirements.txt
(Python version is the 3.10 and the GPU is the V100 with cuda 12.1)
```
### Prepare Datasets
Download the datasets [IWSLT-2017](https://huggingface.co/datasets/IWSLT/iwslt2017/tree/main)

Then rename them under the directory like follow:

```
data
├── IWSLT2017
│   │── data
│   │   └── 2017-01-trnted
│   │       └── texts
│   │           └── en
│   │               └── de
│   │                   └── en-de.zip
│   └── iwslt2017.py
```

## Training
We utilize 1 GPU for training with 32G of memory.

### Scripts.
Command input paradigm
`bash scripts/run_base.sh`

### Random Seed.
Use `args.seed=42` by default

## Evaluation
Command input paradigm
`bash scripts/eval.sh`
