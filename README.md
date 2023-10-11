## TRS ensemble training code repo

This code repo contains codebase for our proposed TRS ensemble training. We also include other STOA baseline code for fair comparison.

#### Empirical ensemble robustness

* ADP Training (https://arxiv.org/abs/1901.08846)
* GAL Training (https://arxiv.org/abs/1901.09981)
* DVERGE Training (https://arxiv.org/abs/2009.14720)
* Adversarial Training (https://arxiv.org/abs/1706.06083)
* **TRS Training** (ours)

`train/Empirical` folder contains corresponding code to construct above robust ensemble models. You can use the command as

`python train/Empirical/train_xxx.py **kwargs`
`python train/Empirical/train_trs.py cifar10 cifar_resnet20  --num-models 5 --coeff 2 --lamda 2 --scale 5 --debug 0`
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
`pip install advertorch`
`pip install tensorboardX`
`pip install tqdm`
`pip install matplotlib`
`cp library-errors-hot-fix/advertorch/fast_adaptive_boundary.py /home/tnguye11/anaconda3/envs/TRS/lib/python3.11/site-packages/advertorch/attacks/fast_adaptive_boundary.py`

`**kwargs` refers to the training parameters which is defined in `utils/Empirical/arguments.py`

`eval/Empirical` folder contains:

* `whitebox/blackbox.py`: Test the whitebox/blackbox attack robustness of the given ensemble model.
* `decison_boundary.py`: Plot the decision boundary figure around the given input instances
* `trans_matrix.py`: Evaluate the adversarial transferability among base models under various attacks.

`utils/Empirical` folder contains:

* `surrogate.py`: Generate blackbox transfer attack instances from the given surrogate ensemble models.
