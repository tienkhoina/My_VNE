
import os
import hydra
from omegaconf import DictConfig

from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config


# =========================================================
# GENERATE DATASET (USING NRMBCEnv)
# =========================================================
def generate_dataset(config: DictConfig):

    # =========================
    # INIT SYSTEM
    # =========================
    system = BaseSystem.from_config(config)

    controller = system.controller
    recorder = system.recorder
    counter = system.counter
    logger = system.logger

    # =========================
    # OUTPUT PATH
    # =========================
    output_path = os.path.join(os.getcwd(), "bc_dataset.txt")
    print(f"🔥 Generate dataset → {output_path}")

    # clear file 1 lần duy nhất
    open(output_path, "w").close()

    # =========================
    # IMPORT ENV (🔥 QUAN TRỌNG)
    # =========================
    from nrm_bc_env import NRMBCEnv   # sửa path nếu cần

    # =========================
    # LOOP QUA TỪNG INSTANCE
    # =========================
    for i, instance in enumerate(system.instances):

        print(f"\n===== Instance {i} =====")

        p_net = instance.p_net
        v_net = instance.v_net

        # nếu controller có state → reset (an toàn)
        if hasattr(controller, "reset"):
            controller.reset()

        # =========================
        # INIT ENV
        # =========================
        env = NRMBCEnv(
            p_net=p_net,
            v_net=v_net,
            controller=controller,
            recorder=recorder,
            counter=counter,
            logger=logger,
            config=config,

            output_path=output_path,
            debug=False
        )

        # =========================
        # RUN 1 EPISODE
        # =========================
        env.run_episode()

    print("✅ Done generate dataset")


# =========================================================
# ENTRY
# =========================================================
@hydra.main(config_path="settings", config_name="main")
def main(config: DictConfig):

    add_simulation_into_config(config)

    generate_dataset(config)


if __name__ == "__main__":
    main()

