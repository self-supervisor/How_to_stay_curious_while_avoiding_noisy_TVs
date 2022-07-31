import numpy as np
import tensorflow as tf

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_two_dims, unflatten_first_dim, getsess, unet


class Dynamics(object):
    def __init__(
        self,
        auxiliary_task,
        predict_from_pixels,
        ama,
        uncertainty_penalty,
        clip_ama,
        abs_ama,
        clip_val,
        reward_scaling,
        feat_dim=None,
        scope="dynamics",
    ):
        self.abs_ama = abs_ama
        self.clip_val = clip_val
        self.reward_scaling = reward_scaling
        self.ama = ama
        self.clip_ama = clip_ama
        self.uncertainty_penalty = uncertainty_penalty
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.obs = self.auxiliary_task.obs
        self.last_ob = self.auxiliary_task.last_ob
        self.ac = self.auxiliary_task.ac
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        if predict_from_pixels:
            self.features = self.get_features(self.obs, reuse=False)
        else:
            self.features = tf.stop_gradient(self.auxiliary_task.features)

        self.out_features = self.auxiliary_task.next_features

        with tf.variable_scope(self.scope + "_loss"):
            self.loss = self.get_loss()

    def get_features(self, x, reuse):
        nl = tf.nn.leaky_relu
        x_has_timesteps = x.get_shape().ndims == 5
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(
                x, nl=nl, feat_dim=self.feat_dim, last_nl=nl, layernormalize=False
            )
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)

        def add_ac(x):
            return tf.concat([x, ac], axis=-1)

        def residual(x):
            res = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)
            res = tf.layers.dense(add_ac(res), self.hidsize, activation=None)
            return x + res

        def forward_predictor(x):
            x = tf.layers.dense(add_ac(x), self.hidsize, activation=tf.nn.leaky_relu)

            for _ in range(4):
                x = residual(x)
            n_out_features = self.out_features.get_shape()[-1].value
            x = tf.layers.dense(add_ac(x), n_out_features, activation=None)
            x = unflatten_first_dim(x, sh)
            return x

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x_copy = tf.identity(x)
            mu = forward_predictor(x)
            log_sigma_squared = forward_predictor(x_copy)
        if self.ama == "true":
            mse = tf.square(mu - tf.stop_gradient(self.out_features))
            if self.abs_ama == "true":
                dynamics_reward = tf.reduce_mean(
                    tf.abs(mse - tf.exp(log_sigma_squared)), axis=[2]
                )
            elif self.abs_ama == "false":
                dynamics_reward = tf.reduce_mean(
                    (mse - tf.exp(log_sigma_squared)), axis=[2]
                )
            if self.clip_ama == "true":
                dynamics_reward = tf.clip_by_value(dynamics_reward, 0, 1e6)
            dynamics_reward *= self.reward_scaling
            dynamics_reward = tf.clip_by_value(
                dynamics_reward, -self.clip_val, self.clip_val
            )
            loss = tf.reduce_mean(
                (
                    tf.exp(-log_sigma_squared) * mse
                    + self.uncertainty_penalty * log_sigma_squared
                ),
                axis=[2],
            )
        elif self.ama == "false":
            mse = tf.square(mu - tf.stop_gradient(self.out_features))
            dynamics_reward = tf.reduce_mean(mse, axis=[2])
            loss = dynamics_reward
        else:
            raise ValueError("Please specify whether to use AMA or not")
        return (
            loss,
            dynamics_reward,
            log_sigma_squared,
        )

    def calculate_loss(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return [
            getsess().run(
                [self.loss[i] for i in range(len(self.loss))],
                {
                    self.obs: ob[sli(i)],
                    self.last_ob: last_ob[sli(i)],
                    self.ac: acs[sli(i)],
                },
            )
            for i in range(n_chunks)
        ]


class UNet(Dynamics):
    def __init__(
        self,
        auxiliary_task,
        predict_from_pixels,
        ama,
        uncertainty_penalty,
        clip_ama,
        abs_ama,
        clip_val,
        reward_scaling,
        feat_dim=None,
        scope="pixel_dynamics",
    ):
        assert isinstance(auxiliary_task, JustPixels)
        assert (
            not predict_from_pixels
        ), "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        self.clip_ama = clip_ama
        self.abs_ama = abs_ama
        self.ama = ama
        self.uncertainty_penalty = uncertainty_penalty
        self.reward_scaling = reward_scaling
        self.clip_val = clip_val
        super(UNet, self).__init__(
            auxiliary_task=auxiliary_task,
            predict_from_pixels=predict_from_pixels,
            feat_dim=feat_dim,
            scope=scope,
            abs_ama=abs_ama,
            clip_ama=clip_ama,
            ama=ama,
            uncertainty_penalty=uncertainty_penalty,
            reward_scaling=reward_scaling,
            clip_val=clip_val,
        )

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [
                        x,
                        ac_four_dim
                        + tf.zeros(
                            [sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value],
                            tf.float32,
                        ),
                    ],
                    axis=-1,
                )

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            mu, log_sigma_squared = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            mu = unflatten_first_dim(mu, sh)
            log_sigma_squared = unflatten_first_dim(log_sigma_squared, sh)
        prediction_pixels = mu * self.ob_std + self.ob_mean
        if self.ama == "true":
            mse = tf.square(mu - 2 * tf.stop_gradient(self.out_features))
            dynamics_reward = tf.reduce_mean(
                (mse - tf.exp(log_sigma_squared)), axis=[2, 3, 4]
            )
            if self.clip_ama == "true":
                dynamics_reward = tf.clip_by_value(dynamics_reward, 0, 1e6)
            loss = tf.reduce_mean(
                (
                    tf.exp(-log_sigma_squared) * (mse)
                    + self.uncertainty_penalty * log_sigma_squared
                ),
                axis=[2, 3, 4],
            )
        elif self.ama == "false":
            mse = tf.square(mu - tf.stop_gradient(self.out_features))
            dynamics_reward = tf.reduce_mean(mse, axis=[2, 3, 4])
            loss = dynamics_reward
        else:
            raise ValueError("Please specify whether to use AMA or not")
        return (
            loss,
            dynamics_reward,
            prediction_pixels,
            log_sigma_squared,
        )

    def calculate_loss(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        return [
            getsess().run(
                [self.loss[i] for i in range(len(self.loss))],
                {
                    self.obs: ob[sli(i)],
                    self.last_ob: last_ob[sli(i)],
                    self.ac: acs[sli(i)],
                },
            )
            for i in range(n_chunks)
        ]
