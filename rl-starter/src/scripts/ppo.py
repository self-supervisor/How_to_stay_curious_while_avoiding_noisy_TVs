from .dictlist import DictList
import numpy
import torch
import torch.nn.functional as F
import numpy as np
from torch_ac.algos.base import BaseAlgo
from .icm import ICM
from .action_stats_logger import ActionStatsLogger
from welford import Welford
from utils.noisy_tv_wrapper import NoisyTVWrapper


class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(
        self,
        envs,
        acmodel,
        autoencoder,
        autoencoder_opt,
        uncertainty,
        noisy_tv,
        curiosity,
        randomise_env,
        uncertainty_budget,
        environment_seed,
        reward_weighting,
        normalise_rewards,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.001,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-8,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
    ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(
            envs,
            acmodel,
            device,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
        )

        shape = (self.num_frames_per_proc, self.num_procs)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.noisy_tv = noisy_tv
        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        self.icm = ICM(
            autoencoder,
            autoencoder_opt,
            uncertainty,
            device,
            self.preprocess_obss,
        )
        self.visitation_counts = np.zeros(
            (self.env.envs[0].width, self.env.envs[0].height)
        )
        self.reward_weighting = reward_weighting
        self.curiosity = curiosity
        self.intrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.uncertainties = torch.zeros(*shape, device=self.device)
        self.novel_states_visited = torch.zeros(*shape, device=self.device)
        self.normalise_rewards = normalise_rewards
        self.intrinsic_reward_buffer = []
        self.action_stats_logger = ActionStatsLogger(self.env.envs[0].action_space.n)
        self.online_variance = Welford()
        self.normalise_rewards = normalise_rewards
        self.env = NoisyTVWrapper(self.env, self.noisy_tv)

    def update_visitation_counts(self, envs):
        """
        updates counts of novel states visited
        """
        for i, env in enumerate(envs):
            if self.visitation_counts[env.agent_pos[0]][env.agent_pos[1]] == 0:
                pass
                # self.agents_to_save.append(i)
            self.visitation_counts[env.agent_pos[0]][env.agent_pos[1]] += 1

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        # 16 threads running in parallel for 8 frames at a time before parameters
        # are updated, so gathers a total 128 frames
        loss = 0
        for i in range(self.num_frames_per_proc):
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(
                        preprocessed_obs, self.memory * self.mask.unsqueeze(1)
                    )
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            obs, extrinsic_reward, done, _ = self.env.step(action)
            reward = extrinsic_reward
            self.update_visitation_counts(self.env.envs)
            self.obss[i] = self.obs
            self.obs = obs
            if self.curiosity == "True":
                (
                    loss,
                    intrinsic_reward,
                    uncertainty,
                ) = self.icm.compute_intrinsic_rewards(self.obss[i], self.obs, action)
                if self.normalise_rewards == "True":
                    intrinsic_reward_numpy = intrinsic_reward.detach().cpu().numpy()
                    self.online_variance.add_all(intrinsic_reward_numpy)
                    intrinsic_reward /= np.sqrt(self.online_variance.var_s)
                intrinsic_reward *= self.reward_weighting
                reward = intrinsic_reward + torch.tensor(reward, dtype=torch.float).to(
                    self.device
                )
                loss = torch.sum(loss)
                self.intrinsic_reward_buffer.append(intrinsic_reward)
                self.action_stats_logger.add_to_log_dicts(
                    action.detach().numpy(), intrinsic_reward.detach().numpy()
                )
                self.icm.update_curiosity_parameters(loss)
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.curiosity == "True":
                self.uncertainties[i] = uncertainty
                self.intrinsic_rewards[i] = intrinsic_reward
            else:
                self.uncertainties[i] = torch.zeros_like(action)
                self.intrinsic_rewards[i] = torch.zeros_like(action)
            self.novel_states_visited[i] = np.count_nonzero(self.visitation_counts)
            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(
                reward, device=self.device, dtype=torch.float
            )
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(
                self.num_procs, device=self.device
            )

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(
                        self.log_episode_reshaped_return[i].item()
                    )
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(
                    preprocessed_obs, self.memory * self.mask.unsqueeze(1)
                )
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = (
                self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            )
            next_value = (
                self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            )
            next_advantage = (
                self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0
            )

            delta = (
                self.rewards[i]
                + self.discount * next_value * next_mask
                - self.values[i]
            )
            self.advantages[i] = (
                delta + self.discount * self.gae_lambda * next_advantage * next_mask
            )

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [
            self.obss[i][j]
            for j in range(self.num_procs)
            for i in range(self.num_frames_per_proc)
        ]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(
                -1, *self.memories.shape[2:]
            )
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        intrinsic_rewards = self.intrinsic_rewards.transpose(0, 1).reshape(-1)
        uncertainties = self.uncertainties.transpose(0, 1).reshape(-1)
        novel_states_visited = self.novel_states_visited.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "uncertainties": uncertainties,
            "intrinsic_rewards": intrinsic_rewards,
            "novel_states_visited": novel_states_visited,
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs :]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs :]
        self.log_num_frames = self.log_num_frames[-self.num_procs :]

        return exps, logs

    def update_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * sb.advantage
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(
                        value - sb.value, -self.clip_eps, self.clip_eps
                    )
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.value_loss_coef * value_loss
                    )

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = (
                    sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.acmodel.parameters()
                    )
                    ** 0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.acmodel.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[
                (indexes + self.recurrence) % self.num_frames_per_proc != 0
            ]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes
