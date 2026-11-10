"""Downloads and prepares the dataset described in a data config, and writes its dataset card.

uv run scripts/prepare_data.py
uv run scripts/prepare_data.py --config experiments/configs/data_6vars.yaml
uv run scripts/prepare_data.py --set push_to_hub=<user>/<dataset>
"""

import argparse
import logging

from nanoweather.config import load_yaml, parse_args_with_config
from nanoweather.processing.prepare import prepare_dataset, write_dataset_card


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = parse_args_with_config(parser, default_config="configs/data.yaml")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = {key: value for key, value in vars(args).items() if key != "config"}
    feature_extractor, num_steps = prepare_dataset(data_config, args.output_dir)
    card_config = load_yaml(getattr(args, "card_config", "configs/dataset_card.yaml"))
    write_dataset_card(args.output_dir, card_config, data_config, feature_extractor, num_steps)

    repo_id = getattr(args, "push_to_hub", None)
    if repo_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=repo_id, repo_type="dataset")
        logging.info(f"Pushed to https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
