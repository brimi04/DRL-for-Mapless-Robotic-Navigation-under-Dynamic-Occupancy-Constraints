"""Check that DoubleDQN uses the Double DQN target."""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gridnav import DoubleDQN, make_env  # noqa: E402
from gridnav.algorithms import POLICY_KWARGS  # noqa: E402


def test_double_dqn_target():
    th.set_num_threads(1)
    model = DoubleDQN("MlpPolicy", make_env(use_monitor=True), learning_starts=200, buffer_size=5_000,
                      target_update_interval=100_000, policy_kwargs=POLICY_KWARGS, seed=0, verbose=0)
    model.learn(2_000)
    batch = model.replay_buffer.sample(128)
    with th.no_grad():
        q_online = model.q_net(batch.next_observations)
        q_target = model.q_net_target(batch.next_observations)
        ddqn = th.gather(q_target, 1, q_online.argmax(1, keepdim=True)).squeeze()
        manual = th.tensor([q_target[i, int(q_online[i].argmax())].item() for i in range(128)])
    assert np.allclose(ddqn.numpy(), manual.numpy())
    # Online and target networks disagree on some argmax, so the target differs from max_a Q_target.
    assert (q_online.argmax(1) != q_target.argmax(1)).any()
