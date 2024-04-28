import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "gymnax"))
sys.path.insert(0, os.path.join(os.getcwd(), "brax"))

import jax

import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import orbax.checkpoint
import gymnax
from purejaxrl.wrappers import LogWrapper
import imageio
from brax.io import image
from datetime import datetime
import itertools
import json

import pickle

import time
import os

from purejaxrl.wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    StaticNormalizeVecObservation,
    ClipAction
)

import wandb

from wandb_helpers import log_dicts_to_wandb
import pickle


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticVictim(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    obs_victim: jnp.ndarray
    info: jnp.ndarray


def make_train(config):

    config["TOTAL_TIMESTEPS"] = int(np.ceil(config["TOTAL_TIMESTEPS"] / config["EP_LENGTH"]) * config["EP_LENGTH"])

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # episode lendgth must be a multiple of num steps for the logging to work properly
    assert config["EP_LENGTH"] % config["NUM_STEPS"] == 0

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    
    env = StaticNormalizeVecObservation(env)

    assert env_params is None

    victim_params, obs_mean, obs_var = get_victim(config)

    env_params = { "static": True, 
                    "mean": obs_mean,
                    "var": obs_var}

    sim_env, sim_env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    sim_env = ClipAction(sim_env)
    sim_env = VecEnv(sim_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return frac


    def train(lambda_illu, lr, ent_coef, clip_eps, max_grad_norm, rng, victim_params, adv_params):

        if config["ENV_NAME"] == "CartPole-v1":
            network_victim = ActorCriticVictim(
                env.action_space(env_params).n, activation=config["ACTIVATION"]
            )
        else:
            network_victim = ActorCriticVictim(
                env.action_space(env_params).shape[0], activation=config["ACTIVATION"]
            )
            
        rng, _rng = jax.random.split(rng)
        network_params_victim = victim_params

        # INIT ADVERSARY 
        obs_shape_adv = (env.observation_space(env_params).shape[0]*3, ) #adversary has three times the input size
        action_shape_adv = env.observation_space(env_params).shape

        network = ActorCritic(
            action_shape_adv[0], activation=config["ACTIVATION"]
        )

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(obs_shape_adv)

        if adv_params is None:
            print("init adversary from scratch")
            network_params = network.init(_rng, init_x)
        else:
            print("init adversary from passed parameters")
            network_params = adv_params

        def get_lr(count):
            return linear_schedule(count) * lr

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=get_lr, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )

        start_time = time.time()

        def _get_obs_adv(obs, obs_victim, obs_victim_expected, done):

            obs_victim = jnp.ones_like(obs_victim)*-1*done[:, None] + obs_victim*(1-done[:, None])
            obs_victim_expected = jnp.ones_like(obs_victim_expected)*-1*done[:, None] + obs_victim_expected*(1-done[:, None])
    
            obs_adv = jnp.concatenate([obs, obs_victim, obs_victim_expected], axis=1)

            return obs_adv


        def _get_expected_obs_victim(rng, env_state, last_obs_victim, last_action_victim, config):
            
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            last_obs_victim_unnorm = env.unnormalize(last_obs_victim, env_state)

            obsv = sim_env.simulate_step(last_obs_victim_unnorm, last_action_victim)

            obsv = env.normalize(obsv, env_state)

            return obsv


        def _env_step(runner_state, unused):
            train_state, env_state, last_obsv, last_obs_adv, last_obs_victim, last_action_victim, last_done, episode_returns_illu, rng = runner_state

            # Compute expected next victim observation
            obs_victim_expected = _get_expected_obs_victim(rng, env_state, last_obs_victim, last_action_victim, config)
            
            # SELECT ACTION
            obs_adv = _get_obs_adv(last_obsv, last_obs_victim, obs_victim_expected, last_done)
            
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, obs_adv)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            obs_victim = last_obsv + action 

            obs_victim_clipped = jnp.clip(obs_victim, sim_env._env._env._env.env.env.init_obs_low, sim_env._env._env._env.env.env.init_obs_high)

            # clip observation if start of episode
            obs_victim = obs_victim_clipped * last_done[:, None] + obs_victim * (1-last_done)[:, None]

            l2_norm = jnp.linalg.norm(obs_victim - obs_victim_expected, axis=1)
            illu_reward = -1 * jnp.multiply(l2_norm, (1-last_done))

            pi_vic, _ = network_victim.apply(jax.lax.stop_gradient(network_params_victim), obs_victim)
            action_victim = pi_vic.sample(seed=_rng)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])

            obsv, env_state, reward_victim, done, info = env.step(rng_step, env_state, action_victim, env_params)

            new_episode_returns_illu = episode_returns_illu + illu_reward
            returned_episode_returns = jnp.multiply(new_episode_returns_illu, done)
            episode_returns_illu = jnp.multiply(new_episode_returns_illu, (1-done))
            
            info["returned_episode_returns_illu"] = returned_episode_returns

            reward = reward_victim * -1 + lambda_illu * illu_reward

            transition = Transition(
                done, action, value, reward, log_prob, obs_adv, obs_victim, info
            )

            runner_state = (train_state, env_state, obsv, obs_adv, obs_victim, action_victim, done, episode_returns_illu, rng)
            return runner_state, transition

        

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                )
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value


        def _loss_fn(params, traj_batch, gae, targets):
                # RERUN NETWORK
                pi, value = network.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-clip_eps, clip_eps)
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = (
                    0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                )

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = (
                    jnp.clip(
                        ratio,
                        1.0 - clip_eps,
                        1.0 + clip_eps,
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - ent_coef * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy)


        def _update_epoch(update_state, unused):

            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss


        def _update_step(runner_state, unused):
            
            # COLLECT TRAJECTORIES
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_obs_adv, last_obs_victim, last_action_victim, last_done, episode_returns_illu, rng = runner_state

            _, last_val = network.apply(train_state.params, last_obs_adv)

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            metric = traj_batch.info

            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, last_obs_adv, last_obs_victim, last_action_victim, last_done, episode_returns_illu, rng)
        
            return runner_state, metric
        

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )


        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])

        obsv, env_state = env.reset(reset_rng, env_params)

        rng, _rng = jax.random.split(rng)
        obs_shape_vic = env.observation_space(env_params).shape
        action_shape_vic = env.action_space(env_params).shape

        done = jnp.bool_(jnp.ones((config["NUM_ENVS"])))
        episode_returns_illu = jnp.zeros((config["NUM_ENVS"]), dtype=np.float32)
        
        obs_victim = jnp.ones((config["NUM_ENVS"], *obs_shape_vic))*-1
        action_victim = jnp.ones((config["NUM_ENVS"], *action_shape_vic), dtype=env.action_space(env_params).dtype)*-1

        obs_adv = _get_obs_adv(obsv, obs_victim, obs_victim, done)

        runner_state = (train_state, 
                        env_state, 
                        obsv, 
                        obs_adv,
                        obs_victim, 
                        action_victim,
                        done,
                        episode_returns_illu,
                        _rng)
        
        def _n_update_steps(runner_state, unused):

            assert config["METRIC_LOG_EVER_N_EP"]*config["EP_LENGTH"] % config["NUM_STEPS"] == 0
            runner_state, metric_ = jax.lax.scan(_update_step, runner_state, None, config["METRIC_LOG_EVER_N_EP"]*config["EP_LENGTH"]//config["NUM_STEPS"]-1)

            runner_state, metric = _update_step(runner_state, None)
            
            metric['returned_episode'] =  metric['returned_episode'][(-1,), :]
            metric['returned_episode_lengths'] =  metric['returned_episode_lengths'][(-1,), :]
            metric['returned_episode_returns'] =  metric['returned_episode_returns'][(-1,), :]
            metric['returned_episode_returns_illu'] =  metric['returned_episode_returns_illu'][(-1,), :] 
            
            timestep = jnp.max(metric["timestep"])*config["NUM_ENVS"]

            jax.debug.print("{t}, timer_per_1Msteps: {speed}, all returned: {all_ret}. min r: {min_r}, mean r: {mean_r}, max r: {max_r}, min ir: {min_ir}, mean ir {mean_ir}, max ir: {max_ir}", 
            t = timestep,
            speed = (time.time()-start_time)*1e6/timestep,
            all_ret = (metric["returned_episode"]==True).all(),
            min_r = jnp.round(jnp.min(metric["returned_episode_returns"]),2), 
            mean_r = jnp.round(jnp.mean(metric["returned_episode_returns"]),2),
            max_r = jnp.round(jnp.max(metric["returned_episode_returns"]),2), 
            min_ir = jnp.round(jnp.min(metric["returned_episode_returns_illu"]),2),
            mean_ir = jnp.round(jnp.mean(metric["returned_episode_returns_illu"]),2),
            max_ir = jnp.round(jnp.max(metric["returned_episode_returns_illu"]),2))

            return runner_state, metric

        assert config["NUM_UPDATES"] % config["METRIC_LOG_EVER_N_EP"]*config["EP_LENGTH"] % config["NUM_STEPS"] == 0
        runner_state, metric = jax.lax.scan(_n_update_steps, runner_state, None, config["NUM_UPDATES"]//(config["METRIC_LOG_EVER_N_EP"]*config["EP_LENGTH"]//config["NUM_STEPS"]))

        # run another batch to get render images
        _, render_traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["RENDER_STEPS"]
        )

        return {"runner_state": runner_state, "metrics": metric, "last_traj_batch": render_traj_batch}

    return train


def get_output_folder(config):

    tag = config['WANDB_TAG']
    timestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    identifier = f"{config['ENV_NAME']}_{tag}_{timestr}"

    output_folder = os.path.join("outputs", identifier)
    os.makedirs(output_folder, exist_ok=False)

    return output_folder


def get_victim(config):

    env_id = config["ENV_NAME"].split("-")[0].lower()

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    fname_ckpt = f"checkpointing_{env_id}_{config['VICTIM_VERSION']}/"
    abs_fpath_ckpt = os.path.join(os.getcwd(), fname_ckpt)

    raw_restored = orbax_checkpointer.restore(abs_fpath_ckpt)

    obs_mean = pickle.load(open(f"normalization_{env_id}_{config['VICTIM_VERSION']}/mean.pkl", "rb"))
    obs_var = pickle.load(open(f"normalization_{env_id}_{config['VICTIM_VERSION']}/var.pkl", "rb"))

    return raw_restored['model'], obs_mean, obs_var


def extract_wandb_log_dicts(metrics, config):

    n_logsteps = config["N_LOGSTEPS"]
    n_logsteps_total = metrics["returned_episode_returns"].shape[-3]
    log_interval = int(n_logsteps_total/n_logsteps)+1

    assert len(metrics["returned_episode_returns"][~metrics["returned_episode"]]) == 0

    dims = metrics["returned_episode"].shape[:-3]
    assert dims == tuple([len(search) for search in config["SEARCHES"]]) # last dim is seed dim
    
    wandb_logs_dicts = []

    all_combinations = itertools.product(*(range(dim_size) for dim_size in dims))
    for current_indices in all_combinations:

        run_log_dict = {"log_dicts": [],
                        "config": [(config['SEARCH_LABELS'][idx], np.round(float(config['SEARCHES'][idx][value]),5)) for idx, value in enumerate(current_indices)],
                        "wandb_tag": config["WANDB_TAG"]}

        # TODO this can be simplified by using mean over axis
        for logstep_to_eval in range(0, n_logsteps_total, log_interval):

            timestamp = int(np.max(metrics["timestep"][...,logstep_to_eval,:,:])*config["NUM_ENVS"])
            slices = list(current_indices) + [logstep_to_eval, slice(None), slice(None)]

            mean_return_value = np.mean(metrics["returned_episode_returns"][tuple(slices)])
            mean_return_value_illu = np.mean(metrics["returned_episode_returns_illu"][tuple(slices)])

            log_dict = {
                "return": mean_return_value,
                "return_illu": mean_return_value_illu,
                "timestamp": timestamp
            }

            run_log_dict["log_dicts"].append(log_dict)

        wandb_logs_dicts.append(run_log_dict)

    return wandb_logs_dicts


def save_wandb_logs(output_folder):

    config = pickle.load(open(get_config_fpath(output_folder, dtype="pkl"), "rb"))
    outs = pickle.load(open(get_outs_fpath(output_folder), "rb"))

    metrics = outs["metrics"]

    log_dicts = extract_wandb_log_dicts(metrics, config)

    log_dicts_to_wandb(log_dicts)


def save_images_as_gif(images, filename):
    images_uint8 = np.array(images).astype('uint8')
    imageio.mimsave(filename, images_uint8, duration=1) 


def render_obss_array(env, obss):
    pipeline_states_true = jax.vmap(env.get_pipeline_state, in_axes=(0))(obss)
    imgs = jax.vmap(image.render_array, in_axes=(None, 0, None, None))(env._env.sys, pipeline_states_true, 256, 256)

    return imgs


def save_gifs_for_given_params(outs, element, config, output_folder, n_episodes_to_render):

    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None

    _, obs_mean, obs_var = get_victim(config)

    # use the first environment
    dones = outs["last_traj_batch"].done[element][:,0]
    obs = outs["last_traj_batch"].obs[element][:,0,:]
    obs_victim = outs["last_traj_batch"].obs_victim[element][:,0,:]

    # take only the environment observation part of the obs vector
    assert obs.shape[1] % 3 == 0
    obs = obs[:, :obs.shape[1]//3]

    # unnormalise observation    
    obs =  np.multiply(obs, np.sqrt(obs_var + 1e-8)) + obs_mean
    obs_victim =  np.multiply(obs_victim, np.sqrt(obs_var + 1e-8)) + obs_mean
    
    # split observations into episodes
    episode_ends = np.where(dones==True)[0]
    obs_seqs = [obs[episode_ends[i]+1:episode_ends[i+1]] for i in range(len(episode_ends)-1)]
    obs_pert_seqs = [obs_victim[episode_ends[i]+1:episode_ends[i+1]] for i in range(len(episode_ends)-1)]

    assert len(obs_seqs) == len(obs_pert_seqs)

    # render images for each episode
    assert n_episodes_to_render <= len(obs_seqs), "n_episodes_to_render must be less than or equal to the number of episodes"

    for i in range(n_episodes_to_render):

        obss = obs_seqs[i]
        obss_pert = obs_pert_seqs[i]

        imgs = render_obss_array(env, obss)
        imgs_pert = render_obss_array(env, obss_pert)

        concat_images = jnp.concatenate((imgs, imgs_pert), axis=2)
        fname = os.path.join(output_folder, f"illu.gif")

        save_images_as_gif(concat_images, fname)


def get_outs_fpath(output_folder):
    return os.path.join(output_folder, "outs.pkl")


def get_adv_parmas_fpath(output_folder):
    return os.path.join(output_folder, "adv_params.pkl")


def get_config_fpath(output_folder, dtype="pkl"):

    if dtype=="pkl":
        return os.path.join(output_folder, "config.pkl")
    elif dtype=="json":
        return os.path.join(output_folder, "config.json")
    else:
        raise NotImplementedError


def train_agent(config):
    # initialise adversary from scratch
    params_adv = None    

    rng = jax.random.PRNGKey(np.prod(config["SEARCHES"][5]))

    train = make_train(config)

    train = jax.vmap(train, in_axes=(None   , None  , None  , None   , None  , 0     , None, None))
    train = jax.vmap(train, in_axes=(None   , None  , None  , None   , 0     , None  , None, None))
    train = jax.vmap(train, in_axes=(None   , None  , None  , 0      , None  , None  , None, None))
    train = jax.vmap(train, in_axes=(None   , None  , 0     , None   , None  , None  , None, None))
    train = jax.vmap(train, in_axes=(None   , 0     , None  , None   , None  , None  , None, None))
    train = jax.vmap(train, in_axes=(0      , None  , None  , None   , None  , None  , None,  None))
    
    train = jax.jit(train) if not config["DEBUG"] else train      

    outs = jax.block_until_ready(train( jnp.array(config["SEARCHES"][0]),
                                        jnp.array(config["SEARCHES"][1]),
                                        jnp.array(config["SEARCHES"][2]),
                                        jnp.array(config["SEARCHES"][3]),
                                        jnp.array(config["SEARCHES"][4]),
                                        jax.random.split(rng, len(config["SEARCHES"][5])), 
                                        params_victim, 
                                        params_adv))
    return outs


def save_gifs(output_folder):

    # load config file from output folder
    config_loaded = pickle.load(open(get_config_fpath(output_folder, dtype="pkl"), "rb"))
    outs_loaded = pickle.load(open(get_outs_fpath(output_folder), "rb"))

    elem_0 = (0, 0, 0, 0, 0, 0)
    elem_1 = (0, 0, 0, 0, 0, 1)
    elem_2 = (0, 0, 0, 0, 0, 2)

    for element_i, element in enumerate([elem_0, elem_1, elem_2]):
        
        output_folder_ind = os.path.join(output_folder, f"element_{element_i}")
        os.makedirs(output_folder_ind, exist_ok=True)
        
        save_gifs_for_given_params(outs_loaded, element, config_loaded, output_folder_ind, 1)


if __name__ == "__main__":
    
    # element = (0, 0, 1, 1, 0)
    config = {
        "SEARCHES": [[100], [1e-4], [0.02] , [0.1] , [0.025], [0, 1, 2]],
        "SEARCH_LABELS": ["lambda_illu", "lr", "ent_coef", "clip_eps", "max_grad_norm", "seed"],
        "NUM_ENVS": 512,
        "TOTAL_TIMESTEPS": 2e8,
        "NUM_STEPS": 10,
        "EP_LENGTH": 300,
        "METRIC_LOG_EVER_N_EP": 1,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "hopper",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "RENDER_STEPS": 700,
        "WANDB_TAG": "sweep-v2-init",
        "VICTIM_VERSION": "v4",
        "N_LOGSTEPS": 50
    }

    if config["DEBUG"] == True:
        print("IN DEBUG MODE")
        config["NUM_ENVS"] = 32
        config["TOTAL_TIMESTEPS"] = 1e4

    elif config["DEBUG"] == "Testing":
        print("IN TESTING MODE")
        config["NUM_ENVS"] = 32
        config["TOTAL_TIMESTEPS"] = 1e4

    else:
        print("IN REGULAR MODE")

    # load parameters for victim policy
    params_victim, _, _ = get_victim(config)

    # generate output folder and save config 
    output_folder = get_output_folder(config)

    # save config file to output folder
    pickle.dump(config, open(get_config_fpath(output_folder, dtype="pkl"), "wb"))
    json.dump(config, open(get_config_fpath(output_folder, dtype="json"), "w"), indent=4)

    # actually train the agent
    outs = train_agent(config)

    # save adversary policy to folder
    pickle.dump(outs["runner_state"][0].params, open(get_adv_parmas_fpath(output_folder), "wb"))

    # save accumulated metrics to folder
    outs["runner_state"] = None
    pickle.dump(outs, open(get_outs_fpath(output_folder), "wb"))

    # log the outputs to wandb
    save_wandb_logs(output_folder)
    
    # save the gifs
    save_gifs(output_folder)
