import argparse
import math
import time

import numpy as np

import initial
import keras


class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, label, max_seconds, report_every_seconds=15.0, report_every_epochs=10):
        super().__init__()
        self.label = label
        self.max_seconds = max_seconds
        self.report_every_seconds = report_every_seconds
        self.report_every_epochs = report_every_epochs
        self.started_at = None
        self.last_report_at = None
        self.first_loss = None
        self.best_val_loss = math.inf

    def on_train_begin(self, logs=None):
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at
        print(f"[{self.label}] training started")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        now = time.monotonic()
        elapsed = now - self.started_at
        loss = logs.get("loss")
        val_loss = logs.get("val_loss")

        if loss is not None and self.first_loss is None:
            self.first_loss = float(loss)
        if val_loss is not None:
            self.best_val_loss = min(self.best_val_loss, float(val_loss))

        due_by_time = (now - self.last_report_at) >= self.report_every_seconds
        due_by_epoch = (epoch + 1) % self.report_every_epochs == 0
        due_by_end = elapsed >= self.max_seconds

        if due_by_time or due_by_epoch or due_by_end:
            parts = [
                f"[{self.label}] epoch={epoch + 1}",
                f"elapsed={elapsed:.1f}s",
            ]
            if loss is not None:
                parts.append(f"loss={float(loss):.4g}")
                if self.first_loss is not None:
                    parts.append(f"loss_delta={float(loss) - self.first_loss:+.4g}")
            if val_loss is not None:
                parts.append(f"val_loss={float(val_loss):.4g}")
                parts.append(f"best_val={self.best_val_loss:.4g}")
            print(" ".join(parts))
            self.last_report_at = now

        if loss is not None and not np.isfinite(float(loss)):
            print(f"[{self.label}] stopping: non-finite loss")
            self.model.stop_training = True
        elif elapsed >= self.max_seconds:
            print(f"[{self.label}] stopping: reached {self.max_seconds:.0f}s budget")
            self.model.stop_training = True


def make_global_data(n_samples, n_subjects, n_trials):
    data = initial.simulator_hierarchical.sample(
        n_samples,
        n_subjects=n_subjects,
        n_trials=n_trials,
    )
    data["sim_data"] = data["sim_data"].reshape(n_samples, n_subjects * n_trials, 2)
    return data


def make_local_data(n_samples, n_trials):
    return initial.simulator_hierarchical.sample(n_samples, n_trials=n_trials)


def summarize_history(label, history):
    metrics = {}
    for key, values in history.history.items():
        metrics[key] = [float(value) for value in values]

    print(f"[{label}] history={metrics}")
    loss = metrics.get("loss", [])
    val_loss = metrics.get("val_loss", [])

    if not loss or not all(np.isfinite(loss)):
        print(f"[{label}] result=FAIL loss is missing or non-finite")
        return

    loss_delta = loss[-1] - loss[0]
    if val_loss and all(np.isfinite(val_loss)):
        val_delta = val_loss[-1] - val_loss[0]
        print(f"[{label}] result=OK finite losses; loss_delta={loss_delta:+.4g}; val_delta={val_delta:+.4g}")
    else:
        print(f"[{label}] result=OK finite training loss; loss_delta={loss_delta:+.4g}")


def fit_workflow(label, workflow, train_data, val_data, args, seconds):
    print(f"[{label}] train sim_data shape={train_data['sim_data'].shape}")
    print(f"[{label}] val sim_data shape={val_data['sim_data'].shape}")
    callback = ProgressCallback(
        label=label,
        max_seconds=seconds,
        report_every_seconds=args.report_every_seconds,
        report_every_epochs=args.report_every_epochs,
    )
    history = workflow.fit_offline(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[callback],
    )
    summarize_history(label, history)


def main():
    parser = argparse.ArgumentParser(description="Short BayesFlow training smoke test.")
    parser.add_argument("--which", choices=["global", "local", "both"], default="both")
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--epochs", type=int, default=100_000)
    parser.add_argument("--n-train", type=int, default=1024)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-subjects", type=int, default=initial.N_LOCAL_SUBJECTS)
    parser.add_argument("--n-trials", type=int, default=initial.N_TRIALS)
    parser.add_argument("--report-every-seconds", type=float, default=15.0)
    parser.add_argument("--report-every-epochs", type=int, default=10)
    args = parser.parse_args()

    selected = ["global", "local"] if args.which == "both" else [args.which]
    seconds_per_workflow = args.minutes * 60.0 / len(selected)

    print(f"Backend: {keras.backend.backend()}")
    print(f"Selected workflows: {', '.join(selected)}")
    print(f"Time budget: {args.minutes:.1f} min total ({seconds_per_workflow:.1f}s each)")
    print(f"Training samples: {args.n_train}; validation samples: {args.n_val}; batch size: {args.batch_size}")

    if "global" in selected:
        started = time.monotonic()
        train = make_global_data(args.n_train, args.n_subjects, args.n_trials)
        val = make_global_data(args.n_val, args.n_subjects, args.n_trials)
        print(f"[global] simulation elapsed={time.monotonic() - started:.1f}s")
        fit_workflow("global", initial.workflow_global, train, val, args, seconds_per_workflow)

    if "local" in selected:
        started = time.monotonic()
        train = make_local_data(args.n_train, args.n_trials)
        val = make_local_data(args.n_val, args.n_trials)
        print(f"[local] simulation elapsed={time.monotonic() - started:.1f}s")
        fit_workflow("local", initial.workflow_local, train, val, args, seconds_per_workflow)


if __name__ == "__main__":
    main()
