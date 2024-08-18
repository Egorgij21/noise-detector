import sys
sys.path.append("src/")

import argparse
from detector import NoiseDetector


parser = argparse.ArgumentParser(description="Process audio files to detect noise.")

parser.add_argument(
    "--metadata_path", 
    type=str, 
    default="data/total_metadata.txt", 
    help="Path to the metadata file."
)
parser.add_argument(
    "--wavs_path", 
    type=str, 
    default="data/wavs", 
    help="Path to the directory containing wav files."
)
parser.add_argument(
    "--result_filepath",
    type=str,
    default="data/conclusion.txt",
    help="Path to the file with final errors."
)
parser.add_argument(
    "--do_check_annotations", 
    action="store_true",
    help="Flag to check annotations."
)
parser.add_argument(
    "--do_check_speech", 
    action="store_true",
    help="Flag to check speech."
)
parser.add_argument(
    "--do_lang_check", 
    action="store_true",
    help="Flag to check language.",
)
parser.add_argument(
    "--do_nisqa_check",
    action="store_true",
    help="Flag to check nisqa signal metrics.",
)
parser.add_argument(
    "--lang", 
    type=str,
    help="Audios speech language.",
    default="en"
)


def main(metadata_path, wavs_path, result_filepath, do_check_annotations, do_check_speech, do_lang_check, do_nisqa_check, lang):
    detector = NoiseDetector(
        metadata_path=metadata_path,
        wavs_path=wavs_path,
        result_filepath=result_filepath,
        do_check_annotations=do_check_annotations,
        do_check_speech=do_check_speech,
        do_nisqa_check=do_nisqa_check,
        do_lang_check=do_lang_check,
        lang=lang
    )

    detector.process_audios(make_dataframe=True)


if __name__ == "__main__":

    args = parser.parse_args()

    dataframe = main(
        metadata_path=args.metadata_path,
        wavs_path=args.wavs_path,
        result_filepath=args.result_filepath,
        do_check_annotations=args.do_check_annotations,
        do_check_speech=args.do_check_speech,
        do_lang_check=args.do_lang_check,
        do_nisqa_check=args.do_nisqa_check,
        lang=args.lang
    )
