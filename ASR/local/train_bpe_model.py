import argparse
import shutil
from pathlib import Path
from typing import Dict

import sentencepiece as spm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        The generated bpe.model is saved to this directory.
    Generate the tokens.txt from a bpe model.
