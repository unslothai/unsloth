import dataclasses
import json
import os

from transformers.trainer_callback import ProgressCallback
from transformers.trainer_pt_utils import _secs2timedelta


class MetricsCallBack(ProgressCallback):
    def metrics_format(self, metrics):
        """
        Reformat Trainer metrics values to a human-readable format

        Args:
            metrics (`Dict[str, float]`):
                The metrics returned from train/evaluate/predict

        Returns:
            metrics (`Dict[str, float]`): The reformatted metrics
        """

        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{ v >> 20 }MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{ int(v) >> 30 }GF"
            elif isinstance(metrics_copy[k], float):
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def save_state(self, output_dir, state):
        json_string = (
            json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n"
        )
        json_path = os.path.join(output_dir, f"state-{state.global_step}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs_formatted = self.metrics_format(logs)
        k_width = max(len(str(x)) for x in logs_formatted.keys())
        v_width = max(len(str(x)) for x in logs_formatted.values())
        print("Global Step: ", state.global_step)
        for key in sorted(logs_formatted.keys()):
            print(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")

    def on_train_end(self, args, state, control, **kwargs):
        self.save_state(args.output_dir, state)
        super().on_train_end(args, state, control, **kwargs)
