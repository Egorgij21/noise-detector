# Noise-Detector

## Intro
First upload your data to the repository, for convenience you can use the data/ directory.
To avoid dependency issues, it is suggested to raise the solution in docker, but you can also use the functions in your local environment.

## Docker
Building and executing:
```
cd docker
docker compose -p noise-reducer -f docker-compose.yml build
docker run -dt --name noisecheck  noise-reducer-noisecheck
docker exec -it noisecheck bash
mv /workdir/NISQA-s/src/utils ./NISQA-s/src/nisqa_utils && python3.10 bugfix_script.py /workdir/NISQA-s/src/nisqa_utils/process_utils.py
```

## Usage
Data processing:
```
python3.10 -m run --metadata_path "data/total_metadata.txt" --wavs_path "data/wavs" --dst_wavs_folder "data/wavs_denoised" --result_filepath "data/conclusion.txt" --lang "en" --do_check_speech --do_check_annotations --do_lang_check --do_nisqa_check --do_voice_check --do_neural_denoising --do_vocals_denoising
```
