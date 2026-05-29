import importlib.util
import sys
from pathlib import Path


def _load_script(name: str, relative_path: str):
    script_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_square_diffusion_network_config_matches_downloaded_checkpoint_shape():
    for module in (
        _load_script("run_baseline_config", "scripts/run_baseline.py"),
        _load_script("train_replan_ppo_config", "scripts/train_replan_ppo.py"),
    ):
        kwargs = module.diffusion_mlp_kwargs(module.TASKS["robomimic_square"])

        assert kwargs["time_dim"] == 32
        assert kwargs["mlp_dims"] == [1024, 1024, 1024]
        assert kwargs["cond_mlp_dims"] == [512, 64]
        assert kwargs["cond_dim"] == 23
        assert kwargs["horizon_steps"] == 4
        assert kwargs["action_dim"] == 7


def test_lift_and_can_diffusion_network_configs_keep_existing_defaults():
    module = _load_script("run_baseline_default_config", "scripts/run_baseline.py")

    for env in ("robomimic_lift", "robomimic_can"):
        kwargs = module.diffusion_mlp_kwargs(module.TASKS[env])

        assert kwargs["time_dim"] == 16
        assert kwargs["mlp_dims"] == [512, 512, 512]
        assert kwargs["cond_mlp_dims"] is None
