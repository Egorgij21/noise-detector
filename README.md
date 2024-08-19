# Noise-Detector

## Table of Contents
1. [Project Structure](#project-structure)
2. [How to Use](#how-to-use)
    - [Docker Building and Executing](#docker-building)
    - [Run Processing](#run-processing)
    - [Command Line Arguments](#command-line-arguments)
3. [Analysis and Enhancement Methods](#analysis-and-enhancement-methods)
    - [Tag Definitions](#tag-definitions)
    - [Filtering Methods](#filtering-methods)
    - [Denoising Methods](#denoising-methods)
4. [Results](#results)

## Project Structure

```plaintext
.
├── data/                         # Directory containing the input and output files
│   ├── wavs/                     # Directory with the original audio files
│   ├── wavs_denoised/            # Directory to save the processed audio files
│   ├── total_metadata.txt        # Metadata file with annotations for each audio file
│   ├── conclusion.txt            # Output file with results of the analysis
│   └── conclusion.csv            # CSV version of the analysis results
├── docker/                       # Directory with docker files and requirements
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── src/                          # Source code directory
│   ├── detector.py               # Script containing the noise detection and analysis logic
│   ├── denoiser.py               # Script containing the audio denoising logic
│   ├── utils.py                  # Utility functions (not provided in this example)
│   └── voice_utils.py            # Utility functions for voice analysis (not provided in this example)
├── run.py                        # Main script to run the analysis and denoising process
└── README.md                     # Project documentation
```

## How to Use
First upload your data to the repository, for convenience you can use the data/ directory.
To avoid dependency issues, it is suggested to raise the solution in docker, but you can also use the functions in your local environment.
### Docker Building and Executing
```
cd docker
docker compose -p noise-reducer -f docker-compose.yml build
docker run -dt --name noisecheck  noise-reducer-noisecheck
docker exec -it noisecheck bash
mv /workdir/NISQA-s/src/utils ./NISQA-s/src/nisqa_utils && python3.10 bugfix_script.py /workdir/NISQA-s/src/nisqa_utils/process_utils.py
```

### Run Processing
To analyze and enhance the audio files, run the `run.py` script from the command line:
```
python3.10 -m run --metadata_path "data/total_metadata.txt" --wavs_path "data/wavs" --dst_wavs_folder "data/wavs_denoised" --result_filepath "data/conclusion.txt" --lang "en" --do_check_speech --do_check_annotations --do_lang_check --do_nisqa_check --do_voice_check --do_neural_denoising --do_vocals_denoising
```
This command will process the audio files, detect any issues, and attempt to enhance them by applying the specified filters and denoising techniques.

### Command Line Arguments
- `--metadata_path`: Path to the metadata file containing annotations for each audio file.
- `--wavs_path`: Path to the directory containing the input WAV audio files.
- `--dst_wavs_folder`: Directory where the processed (denoised) audio files will be saved.
- `--result_filepath`: Path to the output file that will contain the analysis results.
- `--do_check_annotations`: Check if the transcription annotations match the audio.
- `--do_check_speech`: Verify if the audio contains speech.
- `--do_lang_check`: Detect and validate the language of the speech in the audio.
- `--do_nisqa_check`: Evaluate the audio quality using NISQA metrics.
- `--do_voice_check`: Check for specific voice characteristics (e.g., soft, loud, whisper).
- `--do_neural_denoising`: Apply neural network-based denoising to the audio.
- `--do_vocals_denoising`: Separate and denoise vocals from background noise.
- `--lang`: Specify the language of the speech in the audio (default is English).

## Analysis and Enhancement Methods

### Tag Definitions

During analysis, each audio file is tagged based on identified issues:

- **OK**: The audio file is clean and suitable for use.
- **NO_SPEECH**: The audio contains no detectable speech.
- **WRONG_ANNOTATION**: The transcription does not match the spoken content.
- **WRONG_LANGUAGE**: The spoken language does not match the expected language.
- **LOW_QUALITY**: General poor audio quality.
- **NISQA_NOISE**: Noise detected according to NISQA metrics.
- **BAD_COLORATION**: Audio coloration issues detected.
- **DISCONTINUITY**: Discontinuity in the audio signal.
- **NISQA_BAD_LOUDNESS**: Inconsistent loudness detected.
- **SOFT_SPEECH**: Speech volume is too low.
- **LOUD_SPEECH**: Speech volume is too high.
- **WHISPER_SPEECH**: Speech is whispered.

### Filtering Methods

The following methods are used to filter out or tag problematic audio files:

- **Speech Detection**: Uses `silero-vad` to detect the presence of speech in the audio.
- **Annotation Matching**: Compares the given annotations with ASR-generated transcriptions.
- **Language Detection**: Utilizes `langdetect` to confirm the language of the spoken content.
- **NISQA Evaluation**: Evaluates the audio signal using NISQA to detect noise, coloration, and discontinuities.
- **Voice Characteristics**: Detects and tags unusual voice characteristics like whispering or loud speech.

### Denoising Methods

- **Neural Denoising**: Uses a pre-trained DNS model to remove noise from the audio.
- **Vocals Separation**: Separates vocals from instrumental/background noise using a model from `audio_separator`.

## Results

The analysis results are saved in the following formats:
- **TXT File**: A plain text file (`conclusion.txt`) listing each audio file along with its corresponding tags.
- **CSV File**: A CSV file (`conclusion.csv`) that provides a more detailed, structured report of the analysis.

The processed (denoised) audio files are saved in the directory specified by `--dst_wavs_folder`.
