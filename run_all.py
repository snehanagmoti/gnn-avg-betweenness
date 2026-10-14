"""
Run the whole pipeline: dataset -> training -> evaluation.

Run:  python run_all.py
      python run_all.py --skip-data     (reuse the existing data/ folder)
"""
import argparse
import importlib

from config import CFG

parser = argparse.ArgumentParser()
parser.add_argument("--skip-data", action="store_true", help="don't regenerate the dataset")
args = parser.parse_args()

steps = [("01_generate_dataset", "STEP 1: dataset"),
         ("02_train", "STEP 2+3: model + training"),
         ("03_evaluate", "STEP 4: evaluation")]
for module, title in steps:
    if args.skip_data and module == "01_generate_dataset":
        continue
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    importlib.import_module(module).main(CFG)
