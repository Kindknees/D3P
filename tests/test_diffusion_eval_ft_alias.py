from model.diffusion.diffusion_eval import DiffusionEval as SourceDiffusionEval
from model.diffusion.diffusion_eval_ft import DiffusionEval


def test_diffusion_eval_ft_aliases_diffusion_eval():
    assert DiffusionEval is SourceDiffusionEval
