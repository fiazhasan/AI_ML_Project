"""
Download Stanford Dogs dataset to data/raw.

Uses official Stanford URLs. Requires a valid User-Agent for the server.
Extracts to data/raw/ with Images/ and list files.
"""

import sys
import urllib.request
import tarfile
from pathlib import Path
import argparse
import shutil

# Official Stanford Dogs dataset URLs
IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOTATIONS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
LISTS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"


def print_msg(msg: str, style: str = "info") -> None:
    """Print a styled message to the console (header, success, error, progress)."""
    if style == "header":
        print("\n" + "=" * 60)
        print(f"  {msg}")
        print("=" * 60)
    elif style == "success":
        print(f"\n  [OK] {msg}")
    elif style == "error":
        print(f"\n  [ERROR] {msg}")
    elif style == "progress":
        print(f"\n  >> {msg}")
    else:
        print(f"  {msg}")


def download_file(url: str, dest_path: str) -> None:
    """Download a file from url to dest_path with a progress indicator."""
    print_msg(f"Downloading: {Path(dest_path).name}", "progress")
    print_msg(f"URL: {url}")

    # Stanford server requires a browser-like User-Agent
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
    urllib.request.install_opener(opener)

    def show_progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            percent = min(100, block_num * block_size * 100 // total_size)
            mb_done = (block_num * block_size) / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  Progress: {percent}% ({mb_done:.1f} / {mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=show_progress)
        print()
        print_msg(f"Download complete. Saved to: {dest_path}", "success")
    except Exception as e:
        print_msg(f"Download failed. Error: {e}", "error")
        raise


def extract_tar(tar_path: str, extract_to: str) -> None:
    """Extract a tar archive to the given directory."""
    print_msg(f"Extracting: {Path(tar_path).name}", "progress")
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(extract_to)
        print_msg(f"Extraction complete. Output: {extract_to}", "success")
    except Exception as e:
        print_msg(f"Extraction failed. Error: {e}", "error")
        raise


def download_stanford_dogs(data_dir: str = "data/raw", download_images: bool = True) -> None:
    """
    Download and extract the Stanford Dogs dataset.

    Saves list files (train/test splits) and optionally images to data_dir.
    Creates data_dir/Images/ with one subfolder per breed.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print_msg("Stanford Dogs Dataset — Download Script", "header")
    print_msg(f"Target directory: {data_path.absolute()}")

    # Skip if dataset already present
    images_dir = data_path / "Images"
    if images_dir.exists() and any(images_dir.iterdir()):
        print_msg("Dataset already present. Images/ contains data.", "success")
        print_msg("To re-download, remove the data/raw directory first.")
        return

    try:
        # Step 1: Download list files (train/test splits)
        print_msg("Step 1/2: Downloading list files (train/test splits)", "progress")
        lists_tar = data_path / "lists.tar"
        if not lists_tar.exists():
            download_file(LISTS_URL, str(lists_tar))
            extract_tar(str(lists_tar), str(data_path))
            lists_tar.unlink()
        else:
            print_msg("lists.tar already exists. Extracting...", "progress")
            extract_tar(str(lists_tar), str(data_path))

        # Step 2: Download images (~757 MB)
        if download_images:
            print_msg("Step 2/2: Downloading images (~757 MB)", "progress")
            images_tar = data_path / "images.tar"
            if not images_tar.exists():
                download_file(IMAGES_URL, str(images_tar))
                print_msg("Extraction may take a few minutes...", "progress")
                extract_tar(str(images_tar), str(data_path))
                images_tar.unlink()
            else:
                print_msg("images.tar already exists. Extracting...", "progress")
                extract_tar(str(images_tar), str(data_path))

        # Verify output structure
        if images_dir.exists():
            num_folders = len([d for d in images_dir.iterdir() if d.is_dir()])
            print_msg(f"Dataset ready. Found {num_folders} breed folders.", "success")
            print_msg("Run training: python scripts/train.py --model efficientnet")
        else:
            print_msg("WARNING: Images/ folder not found. Server structure may differ.", "error")
            print_msg("Manual download: http://vision.stanford.edu/aditya86/ImageNetDogs/")
            print_msg(f"Extract into: {data_path.absolute()}")
            print_msg("Expected structure: data/raw/Images/n02085620-Chihuahua/ ...")

    except urllib.error.URLError as e:
        print_msg("Network/URL error. Check connectivity or try again later.", "error")
        print_msg(f"Details: {e}", "error")
        print_msg("Alternative — manual download:", "progress")
        print_msg("1. Open: http://vision.stanford.edu/aditya86/ImageNetDogs/")
        print_msg("2. Download: images.tar (757 MB)")
        print_msg("3. Extract to: " + str(data_path.absolute()))
        print_msg("4. Ensure: data/raw/Images/n02085620-Chihuahua/ ...")
        sys.exit(1)
    except Exception as e:
        print_msg(f"Unexpected error: {e}", "error")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Stanford Dogs dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory to save dataset")
    parser.add_argument("--images-only", action="store_true", help="Skip lists; only (re-)download images")
    parser.add_argument("--skip-images", action="store_true", help="Download only list files (for testing)")
    args = parser.parse_args()

    download_stanford_dogs(args.data_dir, download_images=not args.skip_images)
