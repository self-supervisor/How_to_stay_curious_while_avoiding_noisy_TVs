# How to Stay Curious while Avoiding Noisy TVs

Each directory contains different experiment code and instructions 
to run the code.

Logging is performed with weights and biases, if you want to get 
logged data you need to either have a weights and biases account 
or comment out the wandb lines.

# Code Acknowledgements 

The [rl-starter-files](https://github.com/lcswillems/rl-starter-files) files were used as a base for the algorithms in the minigrid experiment. Furthermore, the underlying RL code of the rl-starter files package [torch-ac](https://github.com/lcswillems/torch-ac) was added into this repo to be altered to add the intrinsic reward bonus. Thanks [Lucas Willems](https://github.com/lcswillems) for your fantastic open source contributions.

Also thanks to the amazing [gym-minigrid](https://github.com/Farama-Foundation/gym-minigrid) repo for providing us with an environment to iterate our ideas quickly. For minigrid make sure to install the singelton (non-procedurally generated) Minigrid repo we provide in the minigrid directory to get the results from the paper.

The retro games code is based almost entirely on the [large scale study of cusiorisity driven learning repo](https://arxiv.org/abs/1808.04355) with some small changes to implement aleatoric uncertainty estimation.

When developing the aleatoric uncertainty quantification code, the following repos were helpful:

https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision

https://github.com/pmorerio/dl-uncertainty 

https://github.com/hmi88/what

When developing the forward prediction architecture the following repos were helpful: 

https://github.com/facebookresearch/impact-driven-exploration

https://github.com/L1aoXingyu/295pytorch-beginner/

Misc:

https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream

# If you find that this code/paper is useful enough for a citation use the following bibtex:

```
@inproceedings{mavor2022stay,
  title={How to Stay Curious while avoiding Noisy TVs using Aleatoric Uncertainty Estimation},
  author={Mavor-Parker, Augustine and Young, Kimberly and Barry, Caswell and Griffin, Lewis},
  booktitle={International Conference on Machine Learning},
  pages={15220--15240},
  year={2022},
  organization={PMLR}
}
```

Also please cite the relevant work this builds on (e.g. Large scale study of curiosity driven learning, gym minigrid, torch-ac etc.)
