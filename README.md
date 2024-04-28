## Illusory Attacks JAX Implementation 

Code for the paper [Illusory Attacks: Information-theoretic detectability matters in adversarial attacks](https://www.robots.ox.ac.uk/vgg/research/illusory-attacks/). 
This Code does not exactly reproduce the results in the paper, which were obtained using [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) and Pytorch.

The new JAX implementation is ~100x faster than the original implementation. 
This Code is based on [PureJaxRL](https://github.com/luchris429/purejaxrl). It uses this [Jaxrender](https://github.com/JoeyTeng/jaxrenderer). 


## Installation

```
conda create --name menv_illujax
conda activate menv_illujax
conda install cuda=12.3 cudnn==8.9.7.29 python==3.10
pip install --upgrade "jax[cuda12_local]==0.4.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install evosax distrax optax flax numpy
pip install imageio wandb
pip install -e jaxrenderer
```

## Training the adversary agent
```
python train_adversary.py
```

## Reference
```
@article{franzmeyer2024illusory,
         title = {Illusory Attacks: Information-theoretic detectability matters in adversarial attacks},
         author = {Franzmeyer, Tim and McAleer, Stephen and Henriques, Jo{\~a}o F and Foerster, Jakob N and Torr, Philip HS and Bibi, Adel and de Witt, Christian Schroeder},
         journal={International Conference on Learning Representations},
         year = {2024}
}
```
