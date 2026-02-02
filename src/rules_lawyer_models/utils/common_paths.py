from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommonPaths:
    model_name: str
    INPUTS_DIR: Path = Path("inputs")
    OUTPUTS_DIR: Path = Path("outputs")
    EXPLORATION_REPORTS_DIR: Path = Path("exploration_reports")
    FRAGMENTS_DIR: Path = Path("fragments")

    @property
    def inputs(self) -> Path:
        return CommonPaths.INPUTS_DIR / self.model_name

    @property
    def outputs(self) -> Path:
        return CommonPaths.OUTPUTS_DIR / self.model_name

    @property
    def exploration_reports(self) -> Path:
        return self.outputs / CommonPaths.EXPLORATION_REPORTS_DIR

    @property
    def fragments(self) -> Path:
        return CommonPaths.FRAGMENTS_DIR

    def ensure_all_dirs_exist(self) -> None:
        self.inputs.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.exploration_reports.mkdir(parents=True, exist_ok=True)
