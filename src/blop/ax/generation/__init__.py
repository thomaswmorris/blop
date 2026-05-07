from ax.generation_strategy.generation_strategy import GenerationStrategy

from .nodes import latent_gp_node, sobol_node


def get_generation_strategy(name):

    if name == "beamline":
        return GenerationStrategy(name="Custom Generation Strategy", nodes=[sobol_node, latent_gp_node])
