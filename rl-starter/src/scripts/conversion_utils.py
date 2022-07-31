import numpy as np


def scale_for_autoencoder(obs, normalise=None):
    obs[:, :, :, 0] *= 50
    obs[:, :, :, 1] *= 25
    obs[:, :, :, 2] *= 125
    if normalise:
        obs /= 255
    return obs


def undo_scale_for_autoencoder(obs, normalise=None):
    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, 0)
    if normalise:
        obs *= 255
    obs[:, :, :, 0] /= 50
    obs[:, :, :, 1] /= 25
    obs[:, :, :, 2] /= 125
    return obs


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get("padder", 10)
    vector[: pad_width[0]] = pad_value
    vector[-pad_width[1] :] = pad_value


def convert_representation_to_rgb(representation):
    if len(representation.shape) == 4:
        representation = np.squeeze(representation, 0)
    representation = representation.astype(np.int8)
    representation = np.interp(
        representation, (representation.min(), representation.max()), (0, 255)
    )
    try:
        representation = np.pad(
            representation,
            pad_width=((0, 0), (1, 1), (1, 1)),
            mode="constant",
            constant_values=255,
        )
    except:
        import pdb

        pdb.set_trace()
    return representation


def convert_obs_to_rgb(obs):
    obs = obs["image"]
    obs = np.swapaxes(obs, 2, 0)
    obs = np.interp(obs, (obs.min(), obs.max()), (0, 255))
    obs = np.pad(
        obs, pad_width=((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=255
    )
    return obs
