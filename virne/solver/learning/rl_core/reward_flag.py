from virne.solver.learning.rl_core.reward_calculator import (
    BaseRewardCalculator,
    RewardCalculatorRegistry,
)


@RewardCalculatorRegistry.register("flag")
class FlagRewardCalculator(BaseRewardCalculator):

    def __init__(self, config):

        super().__init__(config)


    def compute(self, p_net, v_net, solution):

        if not solution["result"]:
            return -1.0

        revenue = solution["v_net_revenue"]
        cost = solution["v_net_cost"]

        violation = solution["v_net_max_single_step_hard_constraint_violation"]

        if cost == 0:
            return 0.0

        return revenue / cost - violation