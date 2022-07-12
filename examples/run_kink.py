import sys, os

import numpy as np
import numpy.random as rnd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import gpflow as gp
from gpflow import settings as gps
from GPt import GPSSM_VCDT, GPSSM_FactorizedLinear, PRSSM
from GPt.transitions import KinkTransitions
from GPt.emissions import GaussianEmissions

import matplotlib
import matplotlib.pyplot as plt


def generate_trajectory(session, tr_fn, T, x1=0., process_noise_std=0.):
    """
    # Let's generate a trajectory from an arbitrary starting point:
    :param T:
    :param x1:
    :param process_noise_std:
    :return:
    """
    tf_input = tf.placeholder(dtype=gps.float_type, shape=[1, 1])
    tf_op = tr_fn.conditional_mean(tf_input)
    traj = np.zeros((T, 1))
    traj[0] = x1
    for t in range(T - 1):
        noise = np.random.randn() * process_noise_std
        feed_dict = {tf_input: traj[[t]]}
        traj[t+1] = session.run(tf_op, feed_dict) + noise
    return traj


def plot_trajectory(observations, all_X, all_F, title='"Kink" Transition Function', colmap='viridis'):
    T = len(observations)
    plt.plot(all_X, all_F, color='r', zorder=0, alpha=0.7)

    cmap = matplotlib.cm.get_cmap(colmap)
    plt.scatter(observations[0], observations[1], color=cmap(0), zorder=2)
    for i in range(1, T-1):
        plt.plot([observations[i - 1, 0], observations[i, 0]], [observations[i, 0], observations[i + 1, 0]],
                 color=cmap(i/(T-2)), zorder=1, label='_nolegend_')
        plt.scatter(observations[i], observations[i + 1], color=cmap(i / (T - 2)), zorder=2)

    mappable = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(1, T-1), cmap=cmap)
    mappable.set_array(np.linspace(0, 1, T-1))
    plt.colorbar(mappable, ax=plt.gca(), label='transition index')

    plt.title(title)
    plt.xlabel('$x_t$')
    plt.ylabel('$x_{t+1}$')
    plt.legend(['Transition Function', 'Initial Transition'])


def plot_data(observations, latents, all_X, all_F):
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].plot(latents, 'r-o')
    ax[0].plot(observations, 'k-o')
    ax[0].legend(['Latent States', 'Noisy Observations'], loc='lower left')
    ax[0].set_xlabel('time-step')
    plot_trajectory(observations=observations, all_X=all_X, all_F=all_F, title=None)
    fig.suptitle('A Noisy Trajectory')
    plt.show()


def generate_data(observation_noise_std):
    rnd.seed(0)
    state_dim = 1
    T = 20
    x1 = 0.5
    process_noise_std = 0.05

    tr_fn = KinkTransitions(dim=state_dim)

    session = tr_fn.enquire_session()

    all_X = np.linspace(-3, 1.1, 100)[:, None]
    X_tf = tf.constant(all_X)
    all_F = session.run(tr_fn.conditional_mean(X_tf))

    latents = generate_trajectory(session=session, tr_fn=tr_fn, T=T, x1=x1, process_noise_std=process_noise_std)
    observations = latents + np.random.randn(T, state_dim) * observation_noise_std

    plot_data(observations=observations, latents=latents, all_X=all_X, all_F=all_F)

    return observations


def get_params(state_dim, obs_dim, observation_noise_std, observations):
    """

    :param state_dim:
    :param obs_dim:
    :param observation_noise_std:
    :param observations:
    :return:
    """

    param_dict = {
        # Required arguments:
        'latent_dim': state_dim,  # latent dimensionality of the data
        'Y': observations,  # the observed sequence (i.e. the data)

        # Optional arguments and default values:
        'inputs': None,  # control inputs (if any)
        'emissions': None,  # the emission model (default: linear transformation plus Gaussian noise)
        'px1_mu': None,  # the Gaussian's prior mean for the initial latent state (default: 0)
        'px1_cov': None,  # the Gaussian's prior covariance for the initial
        # latent state (default: identity)
        'kern': None,  # the Gaussian process' kernel (default: Matern 3/2 kernel)
        'Z': None,  # the inducing inputs (default: standard normal samples)
        'n_ind_pts': 100,  # the number of inducing points (ignored if Z is given)
        'mean_fn': None,  # the Gaussian process' mean function (default: the identity function)
        'Q_diag': None,  # the diagonal of the Gaussian process noise's covariance matrix (default: 1)
        'Umu': None,  # the mean of the Gaussian posterior over inducing outputs (default: 0)
        'Ucov_chol': None,  # Cholesky of the covariance matrix of the Gaussian
        # posterior over inducing outputs (default: identity - whitening in use)
        'n_samples': 100,  # number of samples from the posterior with which we will compute the ELBO
        'seed': None,  # random seed for the samples
        'jitter': gps.numerics.jitter_level,  # amount of jitter to be added to the kernel matrix
        'name': None  # the name of the initialised model in the tensorflow graph
    }

    param_dict['Z'] = observations.copy()
    param_dict['mean_fn'] = gp.mean_functions.Zero()

    # Let's also initialise and fix the observation model to the true one.
    # We don't want to learn this here since there's too little data and scale invariances
    # between the GP kernel and the emission's linear transformation. We fix it to the identity
    # plus Gaussian noise:
    obs_noise_covariance = np.eye(obs_dim) * (observation_noise_std ** 2.)
    emissions = GaussianEmissions(
        obs_dim=obs_dim,
        R=obs_noise_covariance)

    emissions.trainable = False
    param_dict['emissions'] = emissions

    return param_dict


def main():
    observation_noise_std = 0.2
    observations = generate_data(observation_noise_std=observation_noise_std)
    param_dict = get_params(state_dim=1, obs_dim=1, observation_noise_std=observation_noise_std,
                            observations=observations)

    vcdt = GPSSM_VCDT(**param_dict)
    optimizer = gp.train.AdamOptimizer()
    maxiter = int(1)
    optimizer.minimize(vcdt, maxiter=maxiter)



if __name__ == "__main__":
    main()




