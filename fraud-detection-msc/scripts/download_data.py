"""
Script to download Credit Card Fraud Detection dataset from Kaggle.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def download_kaggle_dataset():
    """Download the Kaggle credit card fraud dataset."""
    # Check for Kaggle credentials
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    if not kaggle_username or not kaggle_key:
        print("ERROR: Kaggle credentials not found.")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY in your .env file")
        print("To get API credentials:")
        print("1. Go to https://www.kaggle.com/")
        print("2. Navigate to Account > API > Create New API Token")
        print("3. Add credentials to .env file")
        sys.exit(1)

    # Set Kaggle credentials
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key

    try:
        import kaggle

        # Create data directory
        data_dir = Path('data/raw')
        data_dir.mkdir(parents=True, exist_ok=True)

        print("Downloading Credit Card Fraud Detection dataset...")
        print("Dataset: mlg-ulb/creditcardfraud")

        # Download dataset
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud',
            path=str(data_dir),
            unzip=True
        )

        print(f"✓ Dataset downloaded successfully to {data_dir}")
        print(f"✓ Files: {list(data_dir.glob('*'))}")

    except ImportError:
        print("ERROR: kaggle package not installed")
        print("Install with: pip install kaggle")
        sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_kaggle_dataset()
