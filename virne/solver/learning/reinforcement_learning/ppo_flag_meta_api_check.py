import copy
import hydra
import numpy as np

from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id


# =========================================================
# POLICY
# =========================================================

def check_policy(policy):

    print("\n===== CHECK POLICY =====")

    print("forward:", hasattr(policy, "forward"))

    try:
        p2 = copy.deepcopy(policy)
        print("deepcopy:", True)
    except Exception as e:
        print("deepcopy:", False, e)

    try:
        params = list(policy.parameters())
        print("parameters:", len(params) > 0)
    except:
        print("parameters:", False)


# =========================================================
# BUFFER
# =========================================================

def check_buffer(buffer):

    print("\n===== CHECK BUFFER =====")

    attrs = [
        "observations",
        "actions",
        "rewards",
        "returns",
        "logprobs",
        "values",
        "dones",
    ]

    for a in attrs:
        print(a, hasattr(buffer, a))

    funcs = [
        "clear",
        "merge",
        "compute_returns_and_advantages",
        "split_with_instance",
        "size",
    ]

    for f in funcs:
        print(f, hasattr(buffer, f))


# =========================================================
# SOLVER META API
# =========================================================

def check_solver_meta(solver):

    print("\n===== CHECK SOLVER META =====")

    attrs = [
        "gamma",
        "gae_lambda",
        "repeat_times",
        "eps_clip",
        "training_epoch_id",
        "buffer",
        "policy",
        "optimizer",
    ]

    for a in attrs:
        print(a, hasattr(solver, a))

    print("criterion_critic:", hasattr(solver, "criterion_critic"))

    print("preprocess_obs:", hasattr(solver, "preprocess_obs"))

    print("learn_with_instance:", hasattr(solver, "learn_with_instance"))

    print("update:", hasattr(solver, "update"))

    print("evaluate_actions:", hasattr(solver, "evaluate_actions"))


# =========================================================
# PREPROCESS
# =========================================================

def check_preprocess(solver):

    print("\n===== CHECK preprocess_obs =====")

    buf = solver.buffer

    if len(buf.observations) == 0:
        print("buffer empty -> skip")
        return

    try:
        x = solver.preprocess_obs(buf.observations, solver.device)
        print("preprocess ok")
    except Exception as e:
        print("preprocess fail", e)


# =========================================================
# learn_with_instance
# =========================================================

def check_learn_with_instance(system, solver):

    print("\n===== CHECK learn_with_instance =====")

    env = system.env

    try:

        instance = env.get_instance()

    except:

        print("cannot get instance -> skip")
        return

    try:

        out = solver.learn_with_instance(instance)

        if isinstance(out, tuple):

            print("return tuple:", len(out))

        else:

            print("return type:", type(out))

    except Exception as e:

        print("learn_with_instance error:", e)


# =========================================================
# PPO loss requirements
# =========================================================

def check_ppo_required(solver):

    print("\n===== CHECK PPO REQUIRED =====")

    attrs = [
        "gamma",
        "gae_lambda",
        "eps_clip",
        "repeat_times",
        "criterion_critic",
    ]

    for a in attrs:
        print(a, hasattr(solver, a))


# =========================================================
# MAIN
# =========================================================

@hydra.main(
    config_path="../../../../settings",
    config_name="main",
    version_base=None,
)
def run(config):

    print("\n===== PREP CONFIG =====")

    if config.experiment.run_id == "auto":
        config.experiment.run_id = generate_run_id()

    add_simulation_into_config(config)

    print("has simulation:", hasattr(config, "simulation"))

    print("\n===== BUILD SYSTEM =====")

    system = BaseSystem.from_config(config)

    solver = system.solver

    print("\n===== CHECK SOLVER =====")

    print(type(solver))
    print(type(solver.policy))
    print(type(solver.buffer))

    print("\n===== CHECK ENV =====")

    Env = solver.InstanceEnv

    print("get_observation:", hasattr(Env, "get_observation"))
    print("step:", hasattr(Env, "step"))
    print("generate_action_mask:", hasattr(Env, "generate_action_mask"))

    # policy

    check_policy(solver.policy)

    # buffer

    check_buffer(solver.buffer)

    # solver meta

    check_solver_meta(solver)

    # ppo

    check_ppo_required(solver)

    # preprocess

    check_preprocess(solver)

    # learn_with_instance

    check_learn_with_instance(system, solver)


if __name__ == "__main__":
    run()