# Separating the World and Ego Models for Self-Driving
## ICLR 2022 Generalizable Policy Learning in the Physical World Workshop submission

This repository contains the code used for the paper submission.
The repo is largely based on [this repo and the paper](https://github.com/Atcold/pytorch-PPUU).

## Set up
To start using the repository, you will need to run the following command at the root dir.
```
pip install . 
```

### Getting the data
In order to obtain the dataset, please follow the instructions from [this repo](https://github.com/Atcold/pytorch-PPUU).

## Training

A crucial component of the system is the forward model. In order to train the model, we used the following command:

## Forward model training
```
python train_fm.py --model_type fm --configs configs/full_dataset.yaml configs/fm.yaml --values training.experiment_name='fm' training.output_dir=<output> training.enable_latent=True training.batch_size=64 training.n_steps=3e7
```

We make the trained checkpoints available [here](https://drive.google.com/drive/folders/13as7l28LaF65xjlnMEoySx0NgiopgdgU?usp=sharing).

## Policy training
You can train CFM-KM policy as follows:
```
python train_policy.py --model_type vanilla_v3 --configs configs/full_dataset.yaml configs/trained_fm.yaml configs/cfm-km.yaml --values training.output_dir=<output> 
```

## Policy Evaluation
You can evaluate the policy using `eval_policy` script:
```
python eval_policy.py --configs configs/eval_full_dataset.yaml --values checkpoint_path=<policy model path>
```

## DFM-KM MPC evaluation
Here, you would use a `eval_mpc.py`. It works similarly to `eval_policy.py`, except it additionally needs the config for MPC.
```
python eval_mpc.py --configs configs/eval_full_dataset.yaml configs/eval_trained_fm.yaml configs/dfm-km-mpc.yaml --values output_dir=<output> 
```

## CFM-KM MPC evaluation
Unfortunately, this script only works with an earlier version of the code, so before running the script, you need to checkout an older branch:

```
git checkout repro_mpc_fm
python eval_mpc_fm.py --configs configs/eval_full_dataset.yaml configs/eval_trained_fm.yaml configs/cfm-km-mpc.yaml --values output_dir=<output>
```

TODO: add details about evaluating with an environment policy.

## Evaluating with an environment model.
All evaluation scripts support usage of a specified policy model to control other vehicles in the scene. You would need to specify what model to use as follows:

```
python eval_policy.py --configs configs/eval_full_dataset.yaml --values output_dir=<output> checkpoint_path=<model path> old_stats_path=<old stats> env_policy_path=<env policy model path>
```
Older model (from PPUU paper) will need a stats file specified in `old_stats_path` parameter, while CFM-KM policy doesn't need that.
Stats files are generated when first loading the dataset, see 'data/dataloader.py'

## Configuration
All scripts use yaml configuration files, with paths specified after `--configs`, followed by individual values set after `--values`.
`eval_full_dataset.yaml` in the commands above sets the path to the dataset, while `trained_fm.yaml` should specify the path to a trained forward model.
Note that different scripts might have different configuration items specifying the dataset path,
therefore you may need separate config files e.g. for training script and for evaluation script.
