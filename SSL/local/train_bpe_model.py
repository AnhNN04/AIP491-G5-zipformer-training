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
        """,
    )

    parser.add_argument(
        "--transcript",
        type=str,
        help="Training transcript.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size for BPE training",
    )

    return parser.parse_args()


def generate_tokens(lang_dir: Path):
    """
    Generate the tokens.txt from a bpe model.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(lang_dir / "bpe.model"))
    token2id: Dict[str, int] = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
    with open(lang_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for sym, i in token2id.items():
            f.write(f"{sym} {i}\n")


def main():
    args = get_args()
    vocab_size = args.vocab_size
    lang_dir = Path(args.lang_dir)

    model_type = "unigram"

    model_prefix = f"{lang_dir}/{model_type}_{vocab_size}"
    train_text = args.transcript
    character_coverage = 1.0
    input_sentence_size = 100000000

    user_defined_symbols = ["<blk>", "<sos/eos>"]
    unk_id = len(user_defined_symbols)
    # Note: unk_id is fixed to 2.
    # If you change it, you should also change other
    # places that are using it.

    model_file = Path(model_prefix + ".model")
    if not model_file.is_file():
        spm.SentencePieceTrainer.train(
            input=train_text,
            vocab_size=vocab_size,
            model_type=model_type,
            model_prefix=model_prefix,
            input_sentence_size=input_sentence_size,
            character_coverage=character_coverage,
            user_defined_symbols=user_defined_symbols,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )
    else:
        print(f"{model_file} exists - skipping")
        return

    shutil.copyfile(model_file, f"{lang_dir}/bpe.model")

    generate_tokens(lang_dir)


if __name__ == "__main__":
    main()
