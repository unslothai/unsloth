import dataclasses
import json
import os
from datetime import datetime

from transformers.trainer_callback import ProgressCallback, TrainerCallback
from transformers.trainer_pt_utils import _secs2timedelta

# Prints filename and line number when logging
LOG_FORMAT_STR = (
    "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
)

TRAINER_PERF_ARGS = {
    "skip_memory_metrics": False,
    "include_num_input_tokens_seen": True,
    "include_tokens_per_second": True,
}


class MetricsCallBack(TrainerCallback):
    def __init__(self, name, verbose=False):
        self.name = name
        self.verbose = verbose

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
        # Format metrics (last entry of log_history)
        log_history = state.log_history
        metrics = self.metrics_format(log_history[-1])
        log_history[-1] = metrics
        state.log_history = log_history

        # Save state
        json_string = (
            json.dumps(dataclasses.asdict(state), indent=2, sort_keys=True) + "\n"
        )

        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = os.path.join(
            output_dir, f"{date_str}-{self.name}-state-{state.global_step}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.verbose:
            logs_formatted = self.metrics_format(logs)
            k_width = max(len(str(x)) for x in logs_formatted.keys())
            v_width = max(len(str(x)) for x in logs_formatted.values())
            print("Global Step: ", state.global_step)
            for key in sorted(logs_formatted.keys()):
                print(f"  {key: <{k_width}} = {logs_formatted[key]:>{v_width}}")
        else:
            return

    def on_train_end(self, args, state, control, **kwargs):
        self.save_state(args.output_dir, state)


#        super().on_train_end(args, state, control, **kwargs)
