from __future__ import annotations

from dataclasses import fields

import pytest

from rules_lawyer_models.training.training_options_factory import TrainingMetaOptions

# ── 1. get_default_sweep_config — structure validation ────────


class TestGetDefaultSweepConfig:
    @pytest.fixture()
    def sweep_config(self) -> dict:
        return TrainingMetaOptions.get_default_sweep_config()

    def test_has_required_top_level_keys(self, sweep_config: dict) -> None:
        assert "method" in sweep_config
        assert "metric" in sweep_config
        assert "parameters" in sweep_config

    def test_method_is_valid(self, sweep_config: dict) -> None:
        assert sweep_config["method"] in ("random", "grid", "bayes")

    def test_metric_has_goal_maximize(self, sweep_config: dict) -> None:
        assert sweep_config["metric"]["goal"] == "maximize"

    def test_metric_has_name(self, sweep_config: dict) -> None:
        assert isinstance(sweep_config["metric"]["name"], str)
        assert len(sweep_config["metric"]["name"]) > 0

    def test_all_meta_option_fields_covered(self, sweep_config: dict) -> None:
        """Every field of TrainingMetaOptions must appear in the sweep parameters."""
        field_names = {f.name for f in fields(TrainingMetaOptions)}
        param_names = set(sweep_config["parameters"].keys())
        assert field_names == param_names

    def test_categorical_params_have_values_key(self, sweep_config: dict) -> None:
        """Categorical parameters must use {"values": [...]} format for W&B."""
        categorical_params = [
            "rank",
            "alpha_multiplier",
            "use_projection_modules",
            "warmup_ratio",
            "lr_schedular_type",
            "optim",
        ]
        for name in categorical_params:
            param = sweep_config["parameters"][name]
            assert "values" in param, f"{name} missing 'values' key"
            assert isinstance(param["values"], list), f"{name} 'values' is not a list"
            assert len(param["values"]) > 0, f"{name} 'values' is empty"

    def test_continuous_params_have_distribution(self, sweep_config: dict) -> None:
        """Continuous parameters must have distribution/min/max for W&B."""
        continuous_params = ["learning_rate", "lora_dropout", "weight_decay"]
        for name in continuous_params:
            param = sweep_config["parameters"][name]
            assert "distribution" in param, f"{name} missing 'distribution'"
            assert "min" in param, f"{name} missing 'min'"
            assert "max" in param, f"{name} missing 'max'"
            assert param["min"] < param["max"], f"{name} min >= max"

    def test_learning_rate_uses_log_uniform(self, sweep_config: dict) -> None:
        lr = sweep_config["parameters"]["learning_rate"]
        assert lr["distribution"] == "log_uniform_values"

    def test_rank_values_are_powers_of_two(self, sweep_config: dict) -> None:
        for v in sweep_config["parameters"]["rank"]["values"]:
            assert v > 0 and (v & (v - 1)) == 0, f"rank value {v} is not a power of 2"

    def test_custom_metric_name_and_goal(self) -> None:
        config = TrainingMetaOptions.get_default_sweep_config(metric_name="accuracy", metric_goal="minimize")
        assert config["metric"]["name"] == "accuracy"
        assert config["metric"]["goal"] == "minimize"

    def test_default_metric_name_is_f1(self) -> None:
        config = TrainingMetaOptions.get_default_sweep_config()
        assert config["metric"]["name"] == "f1"
        assert config["metric"]["goal"] == "maximize"


# ── 2. TrainingMetaOptions.from_dict — sweep-style config ────


class TestTrainingMetaOptionsFromDict:
    def test_round_trip_via_dict(self) -> None:
        """to_dict -> from_dict should produce an equivalent object."""
        original = TrainingMetaOptions.get_simple_default()
        rebuilt = TrainingMetaOptions.from_dict(original.to_dict())
        assert rebuilt == original

    def test_from_dict_with_sweep_style_values(self) -> None:
        """Simulates what wandb.config would provide during a sweep trial."""
        config = {
            "rank": 32,
            "alpha_multiplier": 2,
            "use_projection_modules": False,
            "lora_dropout": 0.05,
            "warmup_ratio": 0.1,
            "learning_rate": 1e-4,
            "optim": "sgd",
            "weight_decay": 0.05,
            "lr_schedular_type": "cosine",
        }
        meta = TrainingMetaOptions.from_dict(config)

        assert meta.rank == 32
        assert meta.alpha_multiplier == 2
        assert meta.use_projection_modules is False
        assert meta.lora_dropout == pytest.approx(0.05)
        assert meta.warmup_ratio == pytest.approx(0.1)
        assert meta.learning_rate == pytest.approx(1e-4)
        assert meta.optim == "sgd"
        assert meta.weight_decay == pytest.approx(0.05)
        assert meta.lr_schedular_type == "cosine"

    def test_from_dict_ignores_extra_keys(self) -> None:
        """wandb.config may contain extra keys (e.g. _wandb metadata)."""
        config = TrainingMetaOptions.get_simple_default().to_dict()
        config["_wandb"] = {"some": "metadata"}
        config["extra_key"] = 42

        meta = TrainingMetaOptions.from_dict(config)
        assert meta == TrainingMetaOptions.get_simple_default()

    def test_from_dict_missing_required_field_raises(self) -> None:
        """If a required field is missing, construction should fail."""
        config = {"rank": 16}  # missing all other fields
        with pytest.raises(TypeError):
            TrainingMetaOptions.from_dict(config)


# ── 3. TrainingMetaOptions.to_training_options ───────────────


class TestToTrainingOptions:
    def test_rank_maps_to_r(self) -> None:
        meta = TrainingMetaOptions.get_simple_default()
        opts = meta.to_training_options()
        assert opts.r == meta.rank

    def test_alpha_is_rank_times_multiplier(self) -> None:
        meta = TrainingMetaOptions(
            rank=16,
            alpha_multiplier=2,
            use_projection_modules=True,
            lora_dropout=0.0,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )
        opts = meta.to_training_options()
        assert opts.lora_alpha == 32

    def test_rslora_enabled_when_rank_above_16(self) -> None:
        meta = TrainingMetaOptions(
            rank=32,
            alpha_multiplier=1,
            use_projection_modules=True,
            lora_dropout=0.0,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )
        assert meta.to_training_options().use_rslora is True

    def test_rslora_disabled_when_rank_16_or_below(self) -> None:
        meta = TrainingMetaOptions(
            rank=16,
            alpha_multiplier=1,
            use_projection_modules=True,
            lora_dropout=0.0,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )
        assert meta.to_training_options().use_rslora is False

    def test_projection_modules_included(self) -> None:
        _meta = TrainingMetaOptions.get_simple_default()
        meta_with = TrainingMetaOptions(
            rank=16,
            alpha_multiplier=1,
            use_projection_modules=True,
            lora_dropout=0.0,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )
        meta_without = TrainingMetaOptions(
            rank=16,
            alpha_multiplier=1,
            use_projection_modules=False,
            lora_dropout=0.0,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_schedular_type="linear",
        )
        opts_with = meta_with.to_training_options()
        opts_without = meta_without.to_training_options()

        assert "gate_proj" in opts_with.target_modules
        assert "gate_proj" not in opts_without.target_modules
        # Both should have attention modules
        assert "q_proj" in opts_with.target_modules
        assert "q_proj" in opts_without.target_modules


# ── 4. SweepCommand instantiation ───────────────────────────


class TestSweepCommandInstantiation:
    def test_default_values(self) -> None:
        from rules_lawyer_models.commands.sweep_command import SweepCommand

        cmd = SweepCommand()
        assert cmd.sweep_count == 5
        assert cmd.sweep_method == "random"

    def test_custom_values(self) -> None:
        from rules_lawyer_models.commands.sweep_command import SweepCommand

        cmd = SweepCommand(sweep_count=10, sweep_method="bayes")
        assert cmd.sweep_count == 10
        assert cmd.sweep_method == "bayes"


# ── 5. Sweep config method override ─────────────────────────


class TestSweepConfigMethodOverride:
    def test_method_can_be_overridden(self) -> None:
        """SweepCommand.execute overrides sweep_config method; verify the pattern works."""
        config = TrainingMetaOptions.get_default_sweep_config()
        config["method"] = "bayes"
        assert config["method"] == "bayes"

    def test_method_can_be_set_to_grid(self) -> None:
        config = TrainingMetaOptions.get_default_sweep_config()
        config["method"] = "grid"
        assert config["method"] == "grid"


# ── 6. run_sweep importability ───────────────────────────────


class TestRunSweepImport:
    def test_run_sweep_importable_from_training(self) -> None:
        from rules_lawyer_models.training import run_sweep

        assert callable(run_sweep)

    def test_run_sweep_importable_from_module(self) -> None:
        from rules_lawyer_models.training.sweep_helper import run_sweep

        assert callable(run_sweep)
