import argparse
import logging
from pathlib import Path

import k2
import kaldifst.utils
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--olabels",
        type=str,
        default=None,
        help="""If not empty, the input FST is assumed to be a transducer
        and we use its attribute specified by "olabels" as the output labels.
        """,
    )
    parser.add_argument(
        "input_filename",
        type=str,
        help="Path to the input FST in k2 format",
    )

    parser.add_argument(
        "output_filename",
        type=str,
        help="Path to the output FST in OpenFst format",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logging.info(f"{vars(args)}")

    input_filename = args.input_filename
    output_filename = args.output_filename
    olabels = args.olabels

    if Path(output_filename).is_file():
        logging.info(f"{output_filename} already exists - skipping")
        return

    assert Path(input_filename).is_file(), f"{input_filename} does not exist"
    logging.info(f"Loading {input_filename}")
    k2_fst = k2.Fsa.from_dict(torch.load(input_filename))
    if olabels:
        assert hasattr(k2_fst, olabels), f"No such attribute: {olabels}"

    p = Path(output_filename).parent
    if not p.is_dir():
        logging.info(f"Creating {p}")
        p.mkdir(parents=True)

    logging.info("Converting (May take some time if the input FST is large)")
    fst = kaldifst.utils.k2_to_openfst(k2_fst, olabels=olabels)
    logging.info(f"Saving to {output_filename}")
    fst.write(output_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
