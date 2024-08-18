import sys
sys.path.append("src/")

import argparse

from detector import NoiseDetector


if __name__ == "__main__":
    detector = NoiseDetector(
        metadata_path="data/total_metadata.txt",
        wavs_path="data/wavs",
        do_check_annotations=False,
        do_check_speech=True
    )

    dataframe = detector.process_audios(return_dataframe=True)
