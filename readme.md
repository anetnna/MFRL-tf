# Requirements
tensorflow > 2.0
magent

# install magent
```shell
cd ./env
./build.sh

vim ~/.zshrc
export PYTHONPATH=./env/python:${PYTHONPATH}
source ~/.zshrc
```

# train in battle
```shell
python train_battle --algo=causal_mfq --map_size=40 --name=causal_mfq_map40
```

the hyperparameter epsilon for CMFQ could be changed in examples/battle_model/causal_inference/infer_utils.py

# train in pursuit
```shell
python train_pursuit --algo=causal_mfq --map_size=40 --name=causal_mfq_map40
```

the hyperparameter epsilon for CMFQ could be changed in examples/pursuit_model/causal_inference/infer_utils.py

# test in battle
```shell
python test_battle --algo=causal_mfq --oppo=mfq --map_size=40 --algo_dir=xxx --oppo_dir=xxx
```

# test in pursuit
```shell
python test_pursuit --algo=causal_mfq --oppo=mfq --map_size=40 --algo_dir=xxx --oppo_dir=xxx
```

