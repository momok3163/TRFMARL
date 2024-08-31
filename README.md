# Code Instruction for TRFMARL

This instruction hosts the PyTorch implementation of TRFMARL accompanying the paper "**A triple rewards framework for multi-agent reinforcement learning**". TRFMARL is a domain-agnostic method that combines the three rewards, namely individual reward, team reward and group reward for efficient cooperation by promoting coordination.

The implementation is based on the frameworks [PyMARL](https://github.com/oxwhirl/pymarl) and [PyMARL2](https://github.com/hijkzzz/pymarl2) with environments [SMAC](https://github.com/oxwhirl/smac). All of our SMAC experiments are based on the latest PyMARL2 utilizing SC2.4.6.10. The underlying dynamics are sufficiently different, so you cannot compare runs across various versions.


## Setup

Conda is needed to create the working environment.

Set up the working environment: 

```shell
bash install_dependencies.sh
pip3 install -r requirements.txt
```

Set up the StarCraftII game core (SC2.4.6.10): 

```shell
bash install_sc2.sh
```


## Training(Requires at least 24GB of Video RAM.)

conda activate trfmarl

To train `TRFMARL` on the SMAC scenarios: 

```shell
python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=5m_vs_6m beta1=10 beta2=.1 anneal_speed=4000000 t_max=10000000 seed=125

python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=3s_vs_5z beta1=1.25 beta2=0.05 anneal_speed=4000000 t_max=10000000 seed=125

python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=corridor beta1=2.5 beta2=0.025 anneal_speed=4000000 t_max=5000000 seed=125 batch_size=64 change_group_batch_size=128

python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=6h_vs_8z beta1=10 beta2=0.1 anneal_speed=4000000 t_max=10000000 seed=125

python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=MMM2 beta1=1.25 beta2=0.025 anneal_speed=4000000 t_max=5000000 seed=125

python3 src/main.py --config=trfmarl --env-config=sc2 with env_args.map_name=2c_vs_64zg beta1=0 beta2=0 anneal_speed=4000000 t_max=5000000 seed=125 batch_size=32 change_group_batch_size=64
```
------

All results will be saved in the `results` folder. 

The config file `src/config/algs/trfmarl.yaml` contains default hyperparameters for TRFMARL.


## Evaluation

### TensorBoard

One could set `use_tensorboard` to `True` in `src/config/default.yaml`, and the training tensorboards will be saved in the `results/tb_logs` directory, containing useful info such as test battle win rate during training. 

### Saving models

Same as PyMARL, set `save_model` to `True` in `src/config/default.yaml`, and the learned model during training will be saved in the `results/models/` directory. The frequency for saving models can be adjusted by setting the parameter `save_model_interval`.

### Loading models

Saved models can be loaded by adjusting the `checkpoint_path` parameter in `src/config/default.yaml`. For instance, to load the model under path `result/model/[timesteps]/agent.th`, set `checkpoint_path` to `result/model/[timesteps]`.

### Saving Starcraft II Replay

The learned model loaded from `checkpoint_path` can be evaluated by setting `evaluate` to `True` in `src/config/default.yaml`. To save the Starcraft II replays, please make sure the configuration `save_replay` is set to `True`, and use the `episode_runner`.

Check out [PyMARL](https://github.com/oxwhirl/pymarl) documentation for more information.

## See Also

See [SMAC](https://github.com/oxwhirl/smac), [PyMARL2](https://github.com/hijkzzz/pymarl2), [PyMARL](https://github.com/oxwhirl/pymarl) for additional instructions.
