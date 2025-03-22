from pathlib import Path
import jsonl

# dataset = "LibriTTS"
# path_base = "/datasets/raw"
#
dataset = "Genshin5.3_JP"
path_base = "/datasets/raw/games_updated/gaming1/Genshin5.3"
#
# dataset = "StarRail2.7_JP"
# path_base = "/datasets/raw/games_updated/gaming1/StarRail2.7"
#
# dataset = "WutheringWaves2.0_JP"
# path_base = "/datasets/raw/games_updated/gaming1/WutheringWaves2.0"
#
dataset_path = Path(f"{path_base}/{dataset}")
audio_suffix = ".wav"
jsonl_path = "/workspace/zaudio_preprocessing/scripts/audiobox_aesthetics/"


# import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

all_audio = [
    p
    for p in dataset_path.glob(f"**/*{audio_suffix}")
    #if stable_id(p.name.split(".", 1)[0]) % LOCAL_RANK_MAX == LOCAL_RANK
]

print(f"Number of {audio_suffix} files in {dataset} = {len(all_audio)}.")

data = []
for audio in all_audio:
    data.append(
        {"path":str(audio)}
    )

jsonl.dump(data, f"{jsonl_path}audiobox_input_{dataset}.jsonl")

