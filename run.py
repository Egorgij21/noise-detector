import sys
sys.path.append("src/")

import re
import argparse
from my_detector import NoiseDetector
from my_denoiser import NoiseReducer


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
    "--dst_wavs_folder", 
    type=str, 
    default="data/wavs_denoised", 
    help="Path to the directory containing denoised wav files."
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
    "--do_voice_check",
    action="store_true",
    help="Flag to check voice peculiarities.",
)
parser.add_argument(
    "--do_neural_denoising",
    action="store_true",
    help="Flag to do neural denoising.",
)
parser.add_argument(
    "--do_vocals_denoising",
    action="store_true",
    help="Flag to separate instrumental and vocals.",
)
parser.add_argument(
    "--lang", 
    type=str,
    help="Audios speech language.",
    default="en"
)


def main(
        metadata_path,
        wavs_path,
        dst_wavs_folder,
        result_filepath,
        do_check_annotations,
        do_check_speech,
        do_lang_check,
        do_nisqa_check,
        do_voice_check,
        do_neural_denoising,
        do_vocals_denoising,
        lang,
        make_dataframe=True
    ):

    detector = NoiseDetector(
        metadata_path=metadata_path,
        wavs_path=wavs_path,
        result_filepath=result_filepath,
        do_check_annotations=do_check_annotations,
        do_check_speech=do_check_speech,
        do_nisqa_check=do_nisqa_check,
        do_lang_check=do_lang_check,
        do_voice_check=do_voice_check,
        lang=lang
    )

    detector.process_audios(make_dataframe=make_dataframe)

    if make_dataframe:
        result_filepath = re.sub(".txt", ".csv", result_filepath)

    denoiser = NoiseReducer(
        wavs_path=wavs_path,
        src_conclusion_filepath=result_filepath,
        do_neural_denoising=do_neural_denoising,
        do_vocals_denoising=do_vocals_denoising
    )

    denoiser.denoise_audios(dst_wavs_folder)


if __name__ == "__main__":

    args = parser.parse_args()

    dataframe = main(
        metadata_path=args.metadata_path,
        wavs_path=args.wavs_path,
        dst_wavs_folder=args.dst_wavs_folder,
        result_filepath=args.result_filepath,
        do_check_annotations=args.do_check_annotations,
        do_check_speech=args.do_check_speech,
        do_lang_check=args.do_lang_check,
        do_nisqa_check=args.do_nisqa_check,
        do_voice_check=args.do_voice_check,
        do_neural_denoising=args.do_neural_denoising,
        do_vocals_denoising=args.do_vocals_denoising,
        lang=args.lang
    )
