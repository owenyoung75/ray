"""A simple multi-agent env with two agents playing rock paper scissors.

This demonstrates running the following policies in competition:
    (1) heuristic policy of repeating the same move
    (2) heuristic policy of beating the last opponent move
    (3) LSTM/feedforward PG policies
    (4) LSTM policy with custom entropy loss
"""

import random
import numpy as np

# from ray.rllib.utils import try_import_tf
# tf = try_import_tf()
import torch

from gym.spaces import Box, Discrete
from ray import tune

from ray.rllib.policy.policy import Policy
# from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.policy.sample_batch import SampleBatch

# from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.pg.pg import PGTrainer

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages



def post_process_advantages(policy, sample_batch, other_agent_batches=None,
                            episode=None):
    """This adds the "advantages" column to the sample train_batch."""
    return compute_advantages(sample_batch, 0.0, policy.config["gamma"],
                              use_gae=False)



# def pg_tf_loss(policy, model, dist_class, train_batch):
#     """The basic policy gradients loss."""
#     logits, _ = model.from_batch(train_batch)
#     action_dist = dist_class(logits, model)
#     return -tf.reduce_mean(action_dist.logp(train_batch[SampleBatch.ACTIONS])
#                            * train_batch[Postprocessing.ADVANTAGES])
#
# PGTFPolicy = build_tf_policy(
#     name="PGTFPolicy",
#     get_default_config=lambda: ray.rllib.agents.pg.pg.DEFAULT_CONFIG,
#     postprocess_fn=post_process_advantages,
#     loss_fn=pg_tf_loss)


def pg_torch_loss(policy, model, dist_class, train_batch):
    """The basic policy gradients loss."""
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    log_probs = action_dist.logp(train_batch[SampleBatch.ACTIONS])
    policy.pi_err = -torch.mean(
        log_probs * train_batch[Postprocessing.ADVANTAGES]
    )
    return policy.pi_err

def pg_loss_stats(policy, train_batch):
    """ The error is recorded when computing the loss."""
    return {"policy_loss": policy.pi_err.item()}

PGTorchPolicy = build_torch_policy(
    name="PGTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.pg.pg.DEFAULT_CONFIG,
    loss_fn=pg_torch_loss,
    stats_fn=pg_loss_stats,
    postprocess_fn=post_process_advantages)



Bound = 10.


class PricingEnv(MultiAgentEnv):
    """Two-player environment for rock paper scissors.

    The observation is simply the last opponent action."""

    def __init__(self, _):
        self.action_space = Box(0., Bound, shape=(1,), dtype=np.float32)
        self.observation_space = Box(0., Bound, shape=(1,), dtype=np.float32)
        self.player1 = "player1"
        self.player2 = "player2"
        self.last_move = None
        self.num_moves = 0

    def reset(self):
        self.last_move = (np.asarray([Bound/2.]), np.asarray([Bound/2.]))
        self.num_moves = 0
        return {self.player1: self.last_move[1], self.player2: self.last_move[0],}

    def step(self, action_dict):
        self.last_move = (action_dict[self.player1], action_dict[self.player2])
        obs = {self.player1: self.last_move[1], self.player2: self.last_move[0],}

        rew = {
            self.player1: int(self.last_move[0]>self.last_move[1]),
            self.player2: int(self.last_move[1]>self.last_move[0]),
        }

        self.num_moves += 1

        done = { "__all__": self.num_moves >= 10,}

        print("\t\t\t\t\t\t", obs)
        return obs, rew, done, {}


class AlwaysSameHeuristic(Policy):
    """Pick a random price and stick with it for the entire episode."""
    def get_initial_state(self):
        mu = Bound/2.
        sigma = 0.2
        return [random.gauss(mu, sigma)]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return list(state_batches[0]), state_batches, {}

    def learn_on_batch(self, samples):
        pass
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass


class BeatLastHeuristic(Policy):
    """Play the move that would beat the last move of the opponent."""
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        return [x+0.05 for x in obs_batch], [], {}

    def learn_on_batch(self, samples):
        pass
    def get_weights(self):
        pass
    def set_weights(self, weights):
        pass


def run_same_policy():
    tune.run("PG", stop={"timesteps_total": 20000}, config={"env": PricingEnv})


def run_heuristic_vs_learned(use_lstm=False, trainer="PG"):
    obs_space = Box(0., Bound, shape=(1,), dtype=np.float32)
    act_space = Box(0., Bound, shape=(1,), dtype=np.float32)

    tune.run(
        trainer,
        stop={"timesteps_total": 10},

        config={
            "env": PricingEnv,

            "gamma": 0.9,
            "num_workers": 4,
            "num_envs_per_worker": 4,
            "sample_batch_size": 10,
            "train_batch_size": 200,

            "multiagent":{
                "policies_to_train": ["learned"],
                "policies": {
                    "always_same": (AlwaysSameHeuristic, obs_space, act_space, {}),
                    "beat_last": (BeatLastHeuristic, obs_space, act_space, {}),
                    "learned": (None, obs_space, act_space, {"model": {"use_lstm": use_lstm}}),
                },
                "policy_mapping_fn": (lambda agent_id: "learned" if agent_id=="player1" else "beat_last"),
            },
        })


def run_with_custom_entropy_loss():
    """Example of customizing the loss function of an existing policy.

    This performs about the same as the default loss does."""

    def entropy_policy_gradient_loss(policy, model, dist_class, train_batch):
        logits, _ = model.from_batch(train_batch)
        action_dist = dist_class(logits, model)
        log_probs = action_dist.logp(train_batch["actions"])

        pi_err = - torch.mean(log_probs * train_batch["advantages"])
        # pi_err = - tf.reduce_mean(log_probs * train_batch["advantages"])
        return -0.1*action_dist.entropy() + pi_err

    # EntropyPolicy = PGTFPolicy.with_updates(loss_fn=entropy_policy_gradient_loss)
    EntropyPolicy = PGTorchPolicy.with_updates(loss_fn=entropy_policy_gradient_loss)

    EntropyLossPG = PGTrainer.with_updates(
        name="EntropyPG",
        get_policy_class=lambda _: EntropyPolicy,
    )
    run_heuristic_vs_learned(use_lstm=True, trainer=EntropyLossPG)


if __name__ == "__main__":
    # run_same_policy()
    # run_heuristic_vs_learned(use_lstm=False)
    run_heuristic_vs_learned(use_lstm=True)
    # run_with_custom_entropy_loss()
