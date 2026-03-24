import torch
from hydra import initialize, compose

from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id

dataset = []

# load config giống main.py
with initialize(version_base=None, config_path="settings"):
    config = compose(config_name="main")

# dùng dataset benchmark đã generate
config.experiment.if_load_p_net = True
config.experiment.if_load_v_nets = True

config.solver.solver_name = "grc_rank"
config.training.use_cuda = False

# fix seed
config.experiment.seed = 0

add_simulation_into_config(config)

system = BaseSystem.from_config(config)

env = system.env
solver = system.solver

obs = env.reset()
done = False

while not done:

    # heuristic solver solve current instance
    solution = solver.solve(obs)

    action = solution["node_mapping"]

    next_obs, reward, done, info = env.step(solution)

    dataset.append({
        "obs": obs,
        "action": action,
        "reward": reward,
        "next_obs": next_obs,
        "done": done
    })

    obs = next_obs

torch.save(dataset, "training_dataset.pt")

print("dataset size:", len(dataset))