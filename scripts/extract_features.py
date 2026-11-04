import argparse
import pandas as pd
from pathlib import Path
from src.data.load_data import load_dataset
from src.features.build_features import build_feature_dataframe

def main():
    parser = argparse.ArgumentParser(description="Extract URL features")
    parser.add_argument("--input", required=True, help="Path to raw CSV with url,label")
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    df_raw = load_dataset(args.input, config_path=args.config)
    df_feat = build_feature_dataframe(df_raw)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(args.output, index=False)
    print(f"Saved features to {args.output} | shape={df_feat.shape}")

if __name__ == "__main__":
    main()
