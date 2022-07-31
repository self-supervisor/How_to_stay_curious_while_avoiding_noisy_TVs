from collections import deque, defaultdict

import numpy as np
from mpi4py import MPI

from utils import getsess
from recorder import Recorder
from record_unwrapped_video import BankHeistVideoRecorder


class Rollout(object):
    def __init__(
        self,
        ob_space,
        ac_space,
        nenvs,
        nsteps_per_seg,
        nsegs_per_env,
        nlumps,
        envs,
        policy,
        int_rew_coeff,
        ext_rew_coeff,
        record_rollouts,
        dynamics,
        args,
    ):
        self.nenvs = nenvs
        self.args = args
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.dynamics = dynamics
        self.reward_fun = (
            lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1.0, 1.0)
            + int_rew_coeff * int_rew
        )

        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty(
            (nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype
        )
        self.buf_obs = np.empty(
            (nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype
        )
        self.buf_predictions = np.empty(
            (nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype
        )
        if self.args["feat_learning"] == "idf":
            self.buf_sigmas = np.empty((nenvs, self.nsteps, 512), np.float32)
        else:
            self.buf_sigmas = np.empty(
                (nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype
            )
        self.buf_obs_last = np.empty(
            (nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32
        )

        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()

        self.env_results = [None] * self.nlumps
        # self.prev_feat = [None for _ in range(self.nlumps)]
        # self.prev_acs = [None for _ in range(self.nlumps)]
        self.int_rew = np.zeros((nenvs,), np.float32)

        self.recorder = (
            Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        )
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []
        self.making_video = False
        self.step_count = 0
        self.rollout_count = 0
        self.action_counts = self.init_action_dict()
        self.intrinsic_rewards_dict = self.init_action_dict()
        if "BankHeist" in self.args["env"]:
            self.bank_heist_recorder = BankHeistVideoRecorder(policy=policy, args=args)

    def init_action_dict(self):
        action_dict = {}
        for action in range(self.envs[0].action_space.n):
            action_dict[action] = 0
        return action_dict

    def get_average_intrinsic_reward_for_each_action(self):
        action_dict = self.init_action_dict()
        for action in range(self.envs[0].action_space.n):
            action_dict[action] = (
                self.intrinsic_rewards_dict[action] / self.action_counts[action]
            )
        return action_dict

    def extract_action_info(self):
        i_len, j_len = self.buf_acs.shape
        for i in range(i_len):
            for j in range(j_len):
                self.action_counts[self.buf_acs[i][j]] += 1
                self.intrinsic_rewards_dict[self.buf_acs[i][j]] += self.buf_rews[i][j]

    def save_video_frames(self):
        from utils import make_mp4

        make_mp4(f"playback/video_tstart_{self.rollout_count}.mp4", self.buf_obs[0])
        make_mp4(
            f"playback/predicted_tstart_{self.rollout_count}.mp4",
            self.buf_predictions[0],
        )
        make_mp4(
            f"playback/sigma_tstart_{self.rollout_count}.mp4",
            self.buf_sigmas[0],
            sigmas=True,
        )
        # np.save(
        #    f"playback/actions_tstart_{self.rollout_count}.npy",
        #    self.buf_acs[0],
        # )

    def collect_rollout(self):
        self.rollout_count += 1
        self.ep_infos_new = []
        for t in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward()
        self.update_info()
        self.extract_action_info()
        # self.update_bank_heist_map()
        print(
            "Average intrinsic rewards",
            self.get_average_intrinsic_reward_for_each_action(),
        )
        print("Action counts", self.action_counts)
        if "BankHeist" in self.args["env"]:
            if self.rollout_count % 10 == 0:
                self.bank_heist_recorder.record_episode()
        # if self.making_video:
        #    self.save_video_frames()

    def calculate_reward(self):
        rewards_and_pixels_and_loss = self.dynamics.calculate_loss(
            ob=self.buf_obs, last_ob=self.buf_obs_last, acs=self.buf_acs
        )
        int_rew = np.array([i[1] for i in rewards_and_pixels_and_loss])
        int_rew = np.concatenate(int_rew, 0)
        if self.args["ama"] == "true":
            sigmas = np.array([i[2] for i in rewards_and_pixels_and_loss])
            sigmas = np.exp(sigmas)
            sigmas = np.concatenate(sigmas, 0)
            self.buf_sigmas[:] = sigmas
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)

    def rollout_step(self):
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        for l in range(self.nlumps):
            obs, prevrews, news, infos = self.env_get(l)
            # if t > 0:
            #     prev_feat = self.prev_feat[l]
            #     prev_acs = self.prev_acs[l]
            for info in infos:
                epinfo = info.get("episode", {})
                mzepinfo = info.get("mz_episode", {})
                retroepinfo = info.get("retro_episode", {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if epinfo:
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    self.ep_infos_new.append((self.step_count, epinfo))

            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)

            acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.env_step(l, acs)

            # self.prev_feat[l] = dyn_feat
            # self.prev_acs[l] = acs
            self.buf_obs[sli, t] = obs
            self.buf_news[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_nlps[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews
            # if t > 0:
            #     dyn_logp = self.policy.call_reward(prev_feat, pol_feat, prev_acs)
            #
            #     int_rew = dyn_logp.reshape(-1, )
            #
            #     self.int_rew[sli] = int_rew
            #     self.buf_rews[sli, t - 1] = self.reward_fun(ext_rew=prevrews, int_rew=int_rew)
            if self.recorder is not None:
                self.recorder.record(
                    timestep=self.step_count,
                    lump=l,
                    acs=acs,
                    infos=infos,
                    int_rew=self.int_rew[sli],
                    ext_rew=prevrews,
                    news=news,
                )
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            for l in range(self.nlumps):
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, _ = self.env_get(l)
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews
                    _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(
                        nextobs
                    )
                    # dyn_logp = self.policy.call_reward(self.prev_feat[l], last_pol_feat, prev_acs)
                    # dyn_logp = dyn_logp.reshape(-1, )
                    # int_rew = dyn_logp
                    #
                    # self.int_rew[sli] = int_rew
                    # self.buf_rews[sli, t] = self.reward_fun(ext_rew=ext_rews, int_rew=int_rew)

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists["eprew"].extend(all_ep_infos["r"])
            self.stats["eprew_recent"] = np.mean(all_ep_infos["r"])
            self.statlists["eplen"].extend(all_ep_infos["l"])
            self.stats["epcount"] += len(all_ep_infos["l"])
            self.stats["tcount"] += sum(all_ep_infos["l"])
            if "visited_rooms" in keys_:
                # Montezuma specific logging.
                self.stats["visited_rooms"] = sorted(
                    list(set.union(*all_ep_infos["visited_rooms"]))
                )
                self.stats["pos_count"] = np.mean(all_ep_infos["pos_count"])
                self.all_visited_rooms.extend(self.stats["visited_rooms"])
                self.all_scores.extend(all_ep_infos["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited rooms")
                    print(self.all_visited_rooms)
                    print("All scores")
                    print(self.all_scores)
            if "levels" in keys_:
                # Retro logging
                temp = sorted(list(set.union(*all_ep_infos["levels"])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited levels")
                    print(self.all_visited_rooms)

            current_max = np.max(all_ep_infos["r"])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out
