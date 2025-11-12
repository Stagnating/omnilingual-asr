#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Download official OmniLingual ASR tokenizers.

This script downloads the SentencePiece tokenizer files used by the
OmniLingual ASR models.

Usage:
    # Download tokenizer for 300m/1b/3b models
    python scripts/download_tokenizer.py --output tokenizer.model

    # Download tokenizer for 7b model
    python scripts/download_tokenizer.py --output tokenizer_v7.model --version v7
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


TOKENIZER_URLS = {
    "default": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model",
    "v7": "https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model",
}

MODEL_TOKENIZER_MAP = {
    "300m": "default",
    "1b": "default",
    "3b": "default",
    "7b": "v7",
    "7b_zs": "default",
}


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from a URL with progress indication.

    Args:
        url: URL to download from
        output_path: Path to save the file
        chunk_size: Size of chunks to download at a time
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        print("Error: urllib is required but not available")
        sys.exit(1)

    print(f"Downloading from {url}")
    print(f"Saving to {output_path}")

    try:
        # Make request
        with urllib.request.urlopen(url) as response:
            # Get file size if available
            file_size = response.headers.get("Content-Length")
            if file_size:
                file_size = int(file_size)
                print(f"File size: {file_size / 1024 / 1024:.2f} MB")

            # Download with progress
            downloaded = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    # Show progress
                    if file_size:
                        percent = (downloaded / file_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB)", end="")
                    else:
                        print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="")

        print("\n✓ Download complete!")

    except urllib.error.HTTPError as e:
        print(f"\n✗ HTTP Error {e.code}: {e.reason}")
        print(f"URL: {url}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\n✗ URL Error: {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


def verify_tokenizer(tokenizer_path: Path) -> bool:
    """Verify that the downloaded tokenizer file is valid.

    Args:
        tokenizer_path: Path to tokenizer file

    Returns:
        True if valid, False otherwise
    """
    if not tokenizer_path.exists():
        return False

    # Check file size (tokenizers should be a few MB)
    file_size = tokenizer_path.stat().st_size
    if file_size < 1000:  # Less than 1KB is definitely wrong
        print(f"Warning: Tokenizer file is suspiciously small ({file_size} bytes)")
        return False

    # Try to load with sentencepiece if available
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
        vocab_size = sp.vocab_size()
        print(f"✓ Tokenizer verified: {vocab_size} tokens")
        return True
    except ImportError:
        print("Note: sentencepiece not installed, skipping detailed verification")
        print("Install with: pip install sentencepiece")
        return True  # Assume valid if we can't verify
    except Exception as e:
        print(f"✗ Tokenizer verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download OmniLingual ASR tokenizers"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer.model"),
        help="Output path for tokenizer file (default: tokenizer.model)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["default", "v7"],
        default="default",
        help="Tokenizer version (default for 300m/1b/3b, v7 for 7b)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        choices=["300m", "1b", "3b", "7b", "7b_zs"],
        help="Model config name (auto-selects correct tokenizer version)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if file exists",
    )

    args = parser.parse_args()

    # Determine version from model config if provided
    if args.model_config:
        args.version = MODEL_TOKENIZER_MAP[args.model_config]
        print(f"Using tokenizer version '{args.version}' for model config '{args.model_config}'")

    # Get URL
    url = TOKENIZER_URLS[args.version]

    # Check if file already exists
    if args.output.exists() and not args.force:
        print(f"Tokenizer file already exists: {args.output}")
        print("Verifying existing file...")
        if verify_tokenizer(args.output):
            print("✓ Existing tokenizer is valid")
            print("\nUse --force to re-download")
            return
        else:
            print("✗ Existing tokenizer appears invalid, re-downloading...")

    # Download
    download_file(url, args.output)

    # Verify
    print("\nVerifying downloaded tokenizer...")
    if verify_tokenizer(args.output):
        print(f"\n✓ Tokenizer ready: {args.output}")
        print(f"  File size: {args.output.stat().st_size / 1024:.1f} KB")
        print(f"\nYou can now use this tokenizer with:")
        print(f"  --tokenizer_path {args.output}")
    else:
        print("\n✗ Downloaded tokenizer may be corrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
