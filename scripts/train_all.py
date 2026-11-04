import argparse
from src.models.train import train_all_models

def main():
    parser = argparse.ArgumentParser(description="Train all SmartPhish models")
    parser.add_argument("--raw", help="Path to raw CSV")
    parser.add_argument("--features", help="Path to processed features CSV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--paths", default="config/paths.yaml")
    args = parser.parse_args()

    results = train_all_models(raw_data_path=args.raw, features_path=args.features, config_path=args.config, paths_path=args.paths)
    print("Training complete. See models/metrics.json")

if __name__ == "__main__":
    main()
