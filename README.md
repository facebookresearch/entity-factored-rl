# Entity Factored RL
This contains code for running the experiments in Policy Architectures for Compositional Generalization in Control.

## Imitation Learning
## Data Collection

Download the weights for generating BC data from [this link](https://drive.google.com/file/d/1XRYpFEHX__SWTyRQjvNG2MHRfSZ94L_N/view?usp=sharing). Unzip the file in `./weights`.
Then generate behavior cloning data by running:
```bash
./script/make_data.sh
```

## BC Training
Sweep over different environments and architectures.
```bash
python launch_bc.py -m +bc_experiment=big_transformer,deepset,mlp +bc_setup=3p,3s,2s2p
```

## Reinforcement Learning
In general, `setup` specifies the environment and exploration schedule and `experiment` specifies the architecture. Some examples:
```bash
# 3 push for transformer and MLP (padded for extrapolation eval)
python launch.py -m +setup=3pdense_fastexp +experiment=transformer,padded_mlp seed="range(5)"

# 3 push for deepset uses a faster exploration schedule
python launch.py -m +setup=3pdense_fastexp +experiment=deepset seed="range(5)"
```

## License
The majority of the code for "Entity Factored RL" is licensed under CC-BY-NC, however portions of the project are available under separate license terms.
* Portions based on [Hindsight Experience Replay](https://github.com/TianhongDai/hindsight-experience-replay), [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3), and [Gym](https://github.com/openai/gym) are licensed under the MIT license.
