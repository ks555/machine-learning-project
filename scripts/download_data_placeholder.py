"""
Placeholder script for downloading phishing datasets.
For research/education, use sources like PhishTank, OpenPhish, or Kaggle.
Always respect terms of service and rate limits.
"""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download placeholder - add your downloader")
    parser.add_argument("--out", default="data/raw/phishing_dataset.csv")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"Placeholder: implement download logic and save to {args.out}")
    print("Suggested columns: url,label (1=phishing,0=legitimate)")

if __name__ == "__main__":
    main()
