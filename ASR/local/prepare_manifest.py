import argparse
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike
from tqdm.auto import tqdm
import random

def prepare_manifest(
    corpus_dir: Pathlike,
    language="vi",
    output_dir: Optional[Pathlike] = None,
    normalize_text: str = "lower",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    audio_dir = corpus_dir / "audio"
    transcripts_dir = corpus_dir / "transcripts"
    assert audio_dir.is_dir(), f"No such directory: {audio_dir}"
    assert transcripts_dir.is_dir(), f"No such directory: {transcripts_dir}"

    wav_files = sorted(list(audio_dir.glob("*.wav")))
    if not wav_files:
        raise ValueError(f"Could not find any .wav files in: {audio_dir}")

    random.seed(42)
    random.shuffle(wav_files)

    num_total = len(wav_files)
    num_dev = int(num_total * 0.05)
    num_test = int(num_total * 0.05)
    num_train = num_total - num_dev - num_test

    splits = {
        "train": wav_files[:num_train],
        "dev": wav_files[num_train:num_train + num_dev],
        "test": wav_files[num_train + num_dev:]
    }

    manifests = {}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests = read_manifests_if_cached(
            dataset_parts=splits.keys(), output_dir=output_dir, prefix="capstone"
        )

    with ThreadPoolExecutor(num_jobs) as ex:
        for part, wav_list in tqdm(splits.items(), desc="Dataset parts"):
            logging.info(f"Processing subset: {part}")
            if manifests_exist(part=part, output_dir=output_dir, prefix="capstone"):
                logging.info(f"Subset: {part} already prepared - skipping.")
                continue
            recordings = []
            supervisions = []
            futures = []
            
            for audio_path in wav_list:
                transcript_path = transcripts_dir / f"{audio_path.stem}.txt"
                futures.append(
                    ex.submit(parse_utterance, audio_path, transcript_path, language)
                )

            for future in tqdm(futures, desc="Processing", leave=False):
                result = future.result()
                if result is None:
                    continue
                recording, segment = result
                recordings.append(recording)
                supervisions.append(segment)

            recording_set = RecordingSet.from_recordings(recordings)
            supervision_set = SupervisionSet.from_segments(supervisions)

            if normalize_text == "lower":
                to_lower = lambda text: text.lower()
                supervision_set = SupervisionSet.from_segments(
                    [s.transform_text(to_lower) for s in supervision_set]
                )

            recording_set, supervision_set = fix_manifests(
                recording_set, supervision_set
            )
            validate_recordings_and_supervisions(recording_set, supervision_set)

            if output_dir is not None:
                supervision_set.to_file(output_dir / f"capstone_supervisions_{part}.jsonl.gz")
                recording_set.to_file(output_dir / f"capstone_recordings_{part}.jsonl.gz")

            manifests[part] = {
                "recordings": recording_set,
                "supervisions": supervision_set,
            }

    return manifests

def parse_utterance(
    audio_path: Path,
    transcript_path: Path,
    language: str,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    if not audio_path.is_file() or not transcript_path.is_file():
        logging.warning(f"Missing file pair: {audio_path} or {transcript_path}")
        return None
    
    recording_id = audio_path.stem
    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    recording = Recording.from_file(audio_path, recording_id=recording_id)
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language=language,
        speaker="unknown",
        text=text,
    )
    return recording, segment

def run(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    lanugage: str,
    normalize_text: str,
    num_jobs: int,
):
    prepare_manifest(
        corpus_dir,
        output_dir=output_dir,
        language=lanugage,
        num_jobs=num_jobs,
        normalize_text=normalize_text,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, help="Path to the data dir.")
    parser.add_argument(
        "--output-dir", type=Path, help="Path where to write the manifests."
    )
    parser.add_argument("--language", type=str, help="dataset language")
    parser.add_argument(
        "--normalize-text",
        type=str,
        help="Conversion of transcripts to lower-case (originally in upper-case)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="How many threads to use (can give good speed-ups with slow disks).",
    )
    args = parser.parse_args()

    run(
        args.corpus_dir,
        args.output_dir,
        args.language,
        args.normalize_text,
        args.num_jobs,
    )
