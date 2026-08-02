"""
This script removes short and long utterances from a cutset.
"""

import argparse
import logging
from pathlib import Path

import sentencepiece as spm
from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import Cut


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=Path,
        help="Path to the bpe.model",
    )

    parser.add_argument(
        "--in-cuts",
        type=Path,
        help="Path to the input cutset",
    )

    parser.add_argument(
        "--out-cuts",
        type=Path,
        help="Path to the output cutset",
    )

    return parser.parse_args()


def filter_cuts(cut_set: CutSet, sp: spm.SentencePieceProcessor):
    total = 0  # number of total utterances before removal
    removed = 0  # number of removed utterances

    def remove_short_and_long_utterances(c: Cut):
        """Return False to exclude the input cut"""
        nonlocal removed, total
        total += 1
        if c.duration < 1.0 or c.duration > 20.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            removed += 1
            return False
        if c.num_frames is None:
            num_frames = c.duration * 100  # approximate
        else:
            num_frames = c.num_frames

        T = ((num_frames - 1) // 2 - 1) // 2

        tokens = sp.encode(c.supervisions[0].text, out_type=str)

        if T < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training. "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].text}. "
                f"Tokens: {tokens}. "
                f"Number of tokens: {len(tokens)}"
            )
            removed += 1
            return False

        return True

    ans = cut_set.filter(remove_short_and_long_utterances).to_eager()
    ratio = removed / total * 100
    logging.info(
        f"Removed {removed} cuts from {total} cuts. {ratio:.3f}% data is removed."
    )
    return ans


def main():
    args = get_args()
    logging.info(vars(args))

    if args.out_cuts.is_file():
        logging.info(f"{args.out_cuts} already exists - skipping")
        return

    assert args.in_cuts.is_file(), f"{args.in_cuts} does not exist"
    assert args.bpe_model.is_file(), f"{args.bpe_model} does not exist"

    sp = spm.SentencePieceProcessor()
    sp.load(str(args.bpe_model))

    cut_set = load_manifest_lazy(args.in_cuts)
    assert isinstance(cut_set, CutSet)

    cut_set = filter_cuts(cut_set, sp)
    logging.info(f"Saving to {args.out_cuts}")
    args.out_cuts.parent.mkdir(parents=True, exist_ok=True)
    cut_set.to_file(args.out_cuts)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    main()
