# Requirements
tensorflow > 2.0

## install MPE
`cd ./MPE` type `pip install -e .`

## Train
`python train_mpe_tag.py --algo=causal_mfq `

## Test
`python test_mpe_tag.py --pred=causal --prey=mfq --pred_dir=xxx --drey_dir=xxx, --idx=149 149`