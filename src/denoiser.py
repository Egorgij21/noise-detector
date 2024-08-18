import os
import pandas as pd

from utils import read_metadata


class NoiseReducer:
     def __init__(
        self,
        metadata_path: str,
        wavs_path: str,
        src_conclusion_filepath: str
    ) -> None:
        assert src_conclusion_filepath.endswith(".csv"), "Specify the path to the recording file in .csv format."

        self.meta = read_metadata(metadata_path)
        self.wavs_path = wavs_path

        self.dataframe = pd.read_csv(src_conclusion_filepath)

        self.POSSIBLE_TAGS = self.dataframe.columns[1:]
