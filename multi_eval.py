import argparse
import glob
import multiprocessing as mp
import os
import subprocess

import ray

import rollout


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",
                        type=str,
                        required=True,
                        help="Base directory; evaluates all runs in subdirs meeting criteria.")
    parser.add_argument("--results_dir",
                        type=str,
                        required=True,
                        help="Directory in which to store results.")
    parser.add_argument("--overwrite",
                        action="store_true",
                        help="If provided, overwrites previous results.")
    parser.add_argument("--num_concurrent",
                        type=int,
                        default=5,
                        help="Number of evaluations to run concurrently.")

    # Arguments for rollout.py.
    parser.add_argument("--algorithm",
                        type=str,
                        default="custom/DataAugmentingPPOTrainer",
                        help="The algorithm or model to train.")
    parser.add_argument("--env",
                        type=str,
                        default="custom_procgen_env_wrapper",
                        help="The gym environment to use.")
    parser.add_argument("--num_episodes",
                        default=10,
                        help="Number of complete episodes to roll out (overrides --steps).")

    return parser


class Run:
    def __init__(self, dirpath, results_dir, checkpoint_filepath):
        assert os.path.exists(dirpath), f"Missing dirpath: {dirpath}"
        self.dirpath = dirpath
        assert os.path.exists(results_dir), f"Missing results dir: {results_dir}"
        self.results_dir = results_dir
        assert os.path.exists(checkpoint_filepath), f"Missing checkpoint: {checkpoint_filepath}"
        self.checkpoint_filepath = checkpoint_filepath

    def evaluate(self, algorithm, env, num_episodes):
        print(f"\nRunning evaluate on run: {self}")
        # # yapf: disable
        # args = [
        #     "--run", algorithm,
        #     "--env", env,
        #     "--episodes", str(num_episodes),
        #     "--out", os.path.join(self.results_dir, "results.out"),
        #     "--no-render",
        #     self.checkpoint_filepath,
        # ]
        # # yapf: enable
        # parser = rollout.create_parser()
        # args = parser.parse_args(args)
        # rollout.run(args, parser)

        print(f"\nRunning evaluate on run: {self}")
        # yapf: disable
        cmd = [
            "rllib",
            "rollout",
            self.checkpoint_filepath,
            "--run", algorithm,
            "--env", env,
            "--episodes", str(num_episodes),
            "--out", os.path.join(self.results_dir, "results.out"),
            "--no-render",
        ]
        # yapf: enable
        subprocess.check_output(cmd)

    @classmethod
    def from_dirpath(cls, dirpath, results_dir, overwrite=False, min_checkpoint=100):
        checkpoint_pattern = os.path.join(dirpath, "**/checkpoint-[0-9][0-9][0-9]")
        checkpoint_filepaths = glob.glob(checkpoint_pattern, recursive=True)
        if len(checkpoint_filepaths) == 0:
            return None

        # Take the last checkpoint.
        checkpoint_filepath = sorted(checkpoint_filepaths)[-1]
        checkpoint_itr = int(checkpoint_filepath.split("checkpoint-")[1])
        if checkpoint_itr < min_checkpoint:
            return None

        if not overwrite and len(os.listdir(results_dir)) > 0:
            return None

        return cls(dirpath, results_dir, checkpoint_filepath)

    def __repr__(self):
        return self.dirpath


class RunEvaluator:
    def __init__(self, base_dir, results_dir, overwrite=False, min_checkpoint=100):
        assert os.path.exists(base_dir), f"Base dir does not exist: {base_dir}"
        self.base_dir = base_dir
        os.makedirs(results_dir, exist_ok=True)
        self.results_dir = results_dir
        self.overwrite = overwrite
        self.min_checkpoint = min_checkpoint
        self.runs = self._load_runs()

    def _load_runs(self):
        run_filename_pattern = os.path.join(self.base_dir, "**/progress.csv")
        filepaths = glob.glob(run_filename_pattern, recursive=True)
        runs = []
        for filepath in filepaths:
            dirpath = filepath.split("progress.csv")[0]
            run_key = dirpath.split(self.base_dir)[1]
            run_results_dir = os.path.join(self.results_dir, run_key)
            os.makedirs(run_results_dir, exist_ok=True)
            run = Run.from_dirpath(dirpath,
                                   run_results_dir,
                                   overwrite=self.overwrite,
                                   min_checkpoint=self.min_checkpoint)
            if run is not None:
                runs.append(run)
        return runs

    def evaluate(self, num_concurrent, *args, **kwargs):
        for run in self.runs:
            run.evaluate(*args, **kwargs)


def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluator = RunEvaluator(args.base_dir, args.results_dir, args.overwrite)
    evaluator.evaluate(args.num_concurrent, args.algorithm, args.env, args.num_episodes)


if __name__ == "__main__":
    main()
