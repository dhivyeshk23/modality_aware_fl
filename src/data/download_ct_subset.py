"""Download a small, reproducible subset of the Mycobacterial CT Imaging Dataset (Kaggle:
damianhan/nifti-dataset) so the CT modality client can train without pulling the full 195GB set.

Usage:
    python -m src.data.download_ct_subset --n-per-class 20
"""
import argparse
import random
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "damianhan/nifti-dataset"
CLASS_PREFIXES = {"TB": "TBNifTI/", "NTM": "NTMNiFTi/"}
SEED = 42


def list_class_files(api: KaggleApi, prefix: str) -> list[str]:
    files, page_token = [], None
    while True:
        resp = (
            api.dataset_list_files(DATASET, page_token=page_token)
            if page_token
            else api.dataset_list_files(DATASET)
        )
        files.extend(f.name for f in resp.files if f.name.startswith(prefix))
        page_token = getattr(resp, "nextPageToken", None) or getattr(resp, "next_page_token", None)
        if not page_token:
            break
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=20)
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parents[2] / "datasets" / "ct_subset"
    )
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()
    rng = random.Random(SEED)

    for label, prefix in CLASS_PREFIXES.items():
        all_files = sorted(list_class_files(api, prefix))
        chosen = rng.sample(all_files, k=min(args.n_per_class, len(all_files)))
        dest = args.out_dir / label
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] {len(all_files)} available, downloading {len(chosen)} -> {dest}")
        for name in chosen:
            local_name = Path(name).name
            if (dest / local_name).exists():
                print(f"  skip (already present): {local_name}")
                continue
            print(f"  downloading: {name}")
            api.dataset_download_file(DATASET, name, path=str(dest), force=False, quiet=False)
            # Kaggle wraps single-file downloads in a zip; unzip in place if needed.
            zip_path = dest / (local_name + ".zip")
            if zip_path.exists():
                import zipfile

                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(dest)
                zip_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
