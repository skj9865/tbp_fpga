import sys
import os

# Add wrapper directory to path so internal imports work
_WRAPPER_DIR = os.path.dirname(os.path.abspath(__file__))
if _WRAPPER_DIR not in sys.path:
    sys.path.insert(0, _WRAPPER_DIR)

# Add scripts/ to path so monty_inference can be imported
_PROJECT_ROOT = os.path.dirname(_WRAPPER_DIR)
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from core.base_algorithm import BaseAlgorithm
from core.registry import register_algorithm


@register_algorithm
class MontyAlgorithm(BaseAlgorithm):
    """Wrapper for Monty inference, compatible with the unified SW framework.

    Expected YAML structure:

        framework:
          device: "cpu"                 # Monty currently runs on CPU only
          seed: 42
          results_dir: "./results"

        algorithms:
          monty:
            enabled: true
            default_dataset: "world_image"
            data_path: null             # null -> use $MONTY_DATA default
            model_path: null            # null -> use $MONTY_MODELS default
            max_episodes: null          # null -> all 48 episodes
            max_eval_steps: 500
            log_level: "INFO"
            output_csv: "eval_stats.csv"

    The framework is expected to flatten framework + algorithm sections into
    a single dict before passing it to configure().
    """

    def __init__(self):
        self._config = {}
        self._dataset = "world_image"
        self._data_path = None
        self._model_path = None
        self._max_episodes = None
        self._max_eval_steps = 500
        self._seed = 42
        self._log_level = "INFO"
        self._output_csv = None
        self._results_dir = None

    def name(self) -> str:
        return "monty"

    def configure(self, config: dict) -> None:
        self._config = config

        # Algorithm-level keys (under `algorithms.monty`)
        self._dataset = config.get("dataset", config.get("default_dataset", "world_image"))
        self._data_path = config.get("data_path")
        self._model_path = config.get("model_path")
        self._max_episodes = config.get("max_episodes")
        self._max_eval_steps = config.get("max_eval_steps", 500)
        self._log_level = config.get("log_level", "INFO")
        self._output_csv = config.get("output_csv")

        # Framework-level keys
        self._seed = config.get("seed", 42)
        self._results_dir = config.get("results_dir")

        # Resolve output_csv against results_dir if it's a relative path
        if self._output_csv and self._results_dir and not os.path.isabs(self._output_csv):
            self._output_csv = os.path.join(self._results_dir, self._output_csv)

    def get_supported_datasets(self) -> list:
        return ["world_image"]

    def _resolve_paths(self):
        """Resolve default paths from monty_inference defaults if not provided."""
        from monty_inference import default_data_path, default_model_path

        data_path = self._data_path or default_data_path()
        model_path = self._model_path or default_model_path()
        return data_path, model_path

    def train(self, **_kwargs) -> dict:
        raise NotImplementedError(
            "MontyAlgorithm is an inference-only wrapper. "
            "Training is not supported through this interface."
        )

    def evaluate(self, **_kwargs) -> dict:
        """Run Monty inference and return aggregate accuracy."""
        original_dir = os.getcwd()
        os.chdir(_PROJECT_ROOT)

        try:
            from monty_inference import run_inference

            data_path, model_path = self._resolve_paths()

            results = run_inference(
                data_path=data_path,
                model_path=model_path,
                max_eval_steps=self._max_eval_steps,
                seed=self._seed,
                log_level=self._log_level,
                output_csv=self._output_csv,
                max_episodes=self._max_episodes,
            )

            total = len(results)
            if total == 0:
                return {"accuracy": 0.0, "dataset": self._dataset, "num_episodes": 0}

            correct = sum(1 for r in results if r["correct"])
            mlh = sum(1 for r in results if r["correct_mlh"])
            combined = correct + mlh

            return {
                "accuracy": combined / total,
                "correct_accuracy": correct / total,
                "mlh_accuracy": mlh / total,
                "num_episodes": total,
                "dataset": self._dataset,
            }
        finally:
            os.chdir(original_dir)
