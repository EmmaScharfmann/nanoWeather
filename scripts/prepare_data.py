"""Prepares the dataset described in a data config and writes its dataset card.

uv run scripts/prepare_data.py
uv run scripts/prepare_data.py --config configs/data.yaml --push_to_hub <user>/<dataset>
"""

import argparse
import logging

from nanoweather.config import load_yaml, parse_args_with_config
from nanoweather.processing.prepare import prepare_dataset, write_dataset_card


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--card_config", type=str, default="configs/dataset_card.yaml")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Dataset repository id.")
    args = parse_args_with_config(parser, default_config="configs/data.yaml")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    data_config = load_yaml(args.config)
    feature_extractor, num_steps = prepare_dataset(data_config, args.output_dir)
    write_dataset_card(args.output_dir, load_yaml(args.card_config), data_config, feature_extractor, num_steps)

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="dataset")
        logging.info(f"Pushed to https://huggingface.co/datasets/{args.push_to_hub}")


if __name__ == "__main__":
    main()
