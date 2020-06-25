import copy
import os
import yaml

import procgen

from train import run, create_parser

ENV_NAMES = ["heist", "caveflyer", "jumper"]
for env_name in ENV_NAMES:
    assert env_name in procgen.env.ENV_NAMES, f"Invalid environment: {env_name}"

def write_experiments(base, num_iterations=5):
    exps = dict()
    for env_name in ENV_NAMES:
        env_dir = os.path.join(base["local_dir"], env_name)
        for iteration in range(num_iterations):
            for ep_adv_weight in [0, 1]:
                base_copy = copy.deepcopy(base)
                base_copy["local_dir"] = env_dir
                base_copy["config"]["env_config"]["env_name"] = env_name
                base_copy["config"]["model"]["custom_options"][
                    "discriminator_weight"] = ep_adv_weight
                exp_name = f"itr_{iteration}_{env_name}_ep_adv_{ep_adv_weight}"
                exps[exp_name] = base_copy

    os.makedirs(base["local_dir"], exist_ok=True)
    exps_filepath = os.path.join(base["local_dir"], "experiments.yaml")
    with open(exps_filepath, "w", encoding="utf-8") as outfile:
        yaml.dump(exps, outfile, default_flow_style=False)
    return exps_filepath


def run_experiments(experiments_filepath):
    parser = create_parser()
    args = parser.parse_args()
    args.config_file = experiments_filepath
    run(args, parser)


def main():
    base_exp_filepath = "./experiments/episode_adversarial.yaml"
    with open(base_exp_filepath) as f:
        base_exp = list(yaml.safe_load(f).values())[0]
    exps_filepath = write_experiments(base_exp)
    run_experiments(exps_filepath)


if __name__ == "__main__":
    main()
