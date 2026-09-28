"""Algorithm construction (A2C, PPO, DQN, DDQN) with the default hyperparameters."""
import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import A2C, PPO, DQN

ALGORITHMS = ["A2C", "PPO", "DQN", "DDQN"]
POLICY_KWARGS = dict(net_arch=[128, 128])  # two hidden layers of 128 units, ReLU

HYPERPARAMS = {
    "A2C": dict(learning_rate=7e-4, n_steps=20, gamma=0.99, gae_lambda=1.0,
                ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5),
    "PPO": dict(learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5,
                max_grad_norm=0.5),
    "DQN": dict(learning_rate=5e-4, buffer_size=100_000, learning_starts=1_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
                exploration_fraction=0.2, exploration_initial_eps=1.0,
                exploration_final_eps=0.05),
}
HYPERPARAMS["DDQN"] = dict(HYPERPARAMS["DQN"])  # identical to DQN; only the TD target differs


class DoubleDQN(DQN):
    """
    Double DQN (van Hasselt et al., 2016) built on Stable-Baselines3 DQN.

    Stable-Baselines3 does not provide DDQN. This class overrides `train()` and
    changes only the TD target: the online network selects the next action and
    the target network evaluates it,
        y = r + gamma * Q_target(s', argmax_a Q_online(s', a)).
    Everything else is identical to the Stable-Baselines3 DQN update.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = getattr(replay_data, "discounts", None)
            if discounts is None:
                discounts = self.gamma
            with th.no_grad():
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = th.gather(self.q_net_target(replay_data.next_observations), 1, next_actions)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
            current_q_values = th.gather(self.q_net(replay_data.observations), 1, replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


ALGO_CLASSES = {"A2C": A2C, "PPO": PPO, "DQN": DQN, "DDQN": DoubleDQN}


def build_model(algo_name, env, overrides=None, verbose=0):
    """Instantiate an algorithm with the default hyperparameters (optionally overridden)."""
    if algo_name not in ALGO_CLASSES:
        raise ValueError(f"Unsupported algorithm: {algo_name}")
    params = dict(HYPERPARAMS[algo_name])
    params.update(overrides or {})
    return ALGO_CLASSES[algo_name](policy="MlpPolicy", env=env,
                                   policy_kwargs=POLICY_KWARGS, verbose=verbose, **params)


def load_model(algo_name, path):
    """Load a saved model (DDQN models load as DoubleDQN; for inference they behave like DQN)."""
    return ALGO_CLASSES[algo_name].load(path, device="cpu")
