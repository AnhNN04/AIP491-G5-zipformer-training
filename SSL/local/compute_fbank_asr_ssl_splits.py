import logging
import multiprocessing
import os
from datetime import datetime
from multiprocessing import Lock, Pool
from pathlib import Path

import torch
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.cut import data

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

device_lock = Lock()


def compute_fbank_asr_ssl_splits(args):
    num_splits = args.num_splits
    output_dir = f"{args.src_dir}/{args.dataset}_split"
    output_dir = Path(output_dir)
    assert output_dir.exists(), f"{output_dir} does not exist!"

    num_digits = 8  

    start = args.start
    stop = args.stop
    if stop < start:
        stop = num_splits

    stop = min(stop, num_splits)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))
    logging.info(f"device: {device}")

    for i in range(start, stop):
        idx = f"{i}".zfill(num_digits)
        logging.info(f"Processing {idx}/{num_splits}")

        cuts_path = output_dir / f"capstone-ssl_cuts_{args.dataset}.{idx}.jsonl.gz"

        raw_cuts_path = (
            output_dir / f"capstone-ssl_cuts_{args.dataset}_raw.{idx}.jsonl.gz"
        )

        logging.info(f"Loading {raw_cuts_path}")
        cut_set = CutSet.from_file(raw_cuts_path)

        logging.info("Computing features")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/capstone-ssl_feats_{idx}",
            num_workers=args.num_workers,
            batch_duration=args.batch_duration,
            overwrite=True,
        )

        logging.info("About to split cuts into smaller chunks.")
        cut_set = cut_set.trim_to_supervisions(
            keep_overlapping=False, min_duration=None
        )

        logging.info(f"Saving to {cuts_path}")
        cut_set.to_file(cuts_path)
        logging.info(f"Saved to {cuts_path}")

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def main():
    import sys
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)

    task = "parallel"
    if len(sys.argv) > 1:
        task = sys.argv[1]

    args = Namespace(
        task=task,
        src_dir="data/ssl_data",
        dataset="ssl",
        num_workers=20,
        batch_duration=600.0,
        num_splits=10,
        start=0,
        stop=-1
    )

    if task == "run":
        for i in range(len(sys.argv)):
            if sys.argv[i] == "--src-dir":
                args.src_dir = sys.argv[i+1]
            elif sys.argv[i] == "--dataset":
                args.dataset = sys.argv[i+1]
            elif sys.argv[i] == "--num-workers":
                args.num_workers = int(sys.argv[i+1])
            elif sys.argv[i] == "--batch-duration":
                args.batch_duration = float(sys.argv[i+1])
            elif sys.argv[i] == "--num-splits":
                args.num_splits = int(sys.argv[i+1])
            elif sys.argv[i] == "--start":
                args.start = int(sys.argv[i+1])
            elif sys.argv[i] == "--stop":
                args.stop = int(sys.argv[i+1])

    if task == "run":
        logging.info(vars(args))
        compute_fbank_asr_ssl_splits(args)
    elif task == "parallel":
        def cleanup(processes):
            print("Cleaning up...")
            for process in processes:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_list = os.environ["CUDA_VISIBLE_DEVICES"]
            device_list = [int(item) for item in device_list.split(",")]
        else:
            device_list = [i for i in range(torch.cuda.device_count())]
        src_dir = args.src_dir
        dataset = args.dataset
        lock_file_name = f"{dataset}_device_lock"

        with open(lock_file_name, "w") as f:
            print(",".join([str(item) for item in device_list]), file=f)

        cuts_lis = os.listdir(os.path.join(src_dir, f"{dataset}_split"))
        cuts_lis = [
            item
            for item in cuts_lis
            if item.endswith("jsonl.gz") and item.find("_raw") >= 0
        ]
        task_list = [int(item.rsplit(".", maxsplit=3)[-3]) for item in cuts_lis]
        os.makedirs("log_tem", exist_ok=True)

        print(f"process number {len(device_list)}")
        process_pool = Pool(len(device_list))
        re_lis = []
        for i, index in enumerate(task_list):
            re_lis.append(
                process_pool.apply_async(run, (src_dir, dataset, index, lock_file_name))
            )
        process_pool.close()
        process_pool.join()
        re_lis = [res.get() for res in re_lis]

def run(src_dir, dataset, index, lock_file_name):
    print(f"task {src_dir} {dataset} {index} start")
    with device_lock:
        with open(lock_file_name, "r") as f:
            line = f.read()
        line = [int(item) for item in line.split(",")]
        print(f"task {src_dir} {dataset} {index} see device {line}")
        assert len(line) > 0
        device = line[0]
        with open(lock_file_name, "w") as f:
            print(",".join([str(item) for item in line[1:]]), file=f)

    print(f"task {src_dir} {dataset} {index} using device {device}")
    state = os.system(
        f"CUDA_VISIBLE_DEVICES={device} PYTHONUTF8=1 python3 ./local/compute_fbank_asr_ssl_splits.py run --src-dir {src_dir} --dataset {dataset} --num-workers 2 --start {index} --stop {index+1} --batch-duration 1000 --num-splits {index+1} 2>&1 | tee log_tem/{dataset}_{index}.log"
    )
    with device_lock:
        with open(lock_file_name, "r") as f:
            line = f.read().strip()
        if len(line) > 0:
            line = [int(item) for item in line.split(",")]
        else:
            line = []
        line.append(device)
        with open(lock_file_name, "w") as f:
            print(",".join([str(item) for item in line]), file=f)
    print(f"task {src_dir} {dataset} {index} finish")
    return state

if __name__ == "__main__":
    main()
