from ax.generation_strategy.generation_strategy import GenerationStrategy

from .nodes import latent_gp_node, sobol_node

all_strategies = ["beamline"]


def get_generation_strategy(name):

    if name not in all_strategies:
        raise ValueError(f"Invalid strategy '{name}', valid strategies are one of {all_strategies}")

    if name == "beamline":
        return GenerationStrategy(name="Custom Generation Strategy", nodes=[sobol_node, latent_gp_node])
