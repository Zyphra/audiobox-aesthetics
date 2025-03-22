## (1). Need to install these in environment.
# pip install audiobox_aesthetics
# pip install py-jsonl
# pip install seaborn


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


## (2). Use make_input_jsonl.py to make a list of audiofiles to compute metric for.

#Sketch-pad: Storage from audiobox_input.jsonl file
#{"path":"/datasets/raw/LibriTTS-R/train_clean_100/LibriTTS_R/train-clean-100/103/1241/103_1241_000060_000003.wav"}
#{"path":"/datasets/raw/LibriTTS-R/train_clean_100/LibriTTS_R/train-clean-100/103/1241/103_1241_000066_000006.wav"}
#{"path":"/datasets/raw/LibriTTS-R/train_clean_100/LibriTTS_R/train-clean-100/103/1241/103_1241_000073_000001.wav"}
#
# {"path":"/datasets/raw/Podcasts/pod_2454710/ep_0.mp3", "start_time":0, "end_time":20}
# {"path":"/datasets/raw/Podcasts/pod_2454710/ep_0.mp3", "start_time":20, "end_time":40}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


## (3). Run in commandline to compute these metrics for a list of audiofiles.
# CUDA_VISIBLE_DEVICES=0 audio-aes audiobox_input_LibriTTS.jsonl --batch-size 100 > audiobox_output_LibriTTS.jsonl
#
# CUDA_VISIBLE_DEVICES=4 audio-aes audiobox_input_Genshin5.3_EN.jsonl --batch-size 100 > audiobox_output_Genshin5.3_EN.jsonl
# CUDA_VISIBLE_DEVICES=5 audio-aes audiobox_input_Genshin5.3_CN.jsonl --batch-size 100 > audiobox_output_Genshin5.3_CN.jsonl
# CUDA_VISIBLE_DEVICES=6 audio-aes audiobox_input_Genshin5.3_KR.jsonl --batch-size 100 > audiobox_output_Genshin5.3_KR.jsonl
# CUDA_VISIBLE_DEVICES=7 audio-aes audiobox_input_Genshin5.3_JP.jsonl --batch-size 100 > audiobox_output_Genshin5.3_JP.jsonl
#
# CUDA_VISIBLE_DEVICES=0 audio-aes audiobox_input_WutheringWaves2.0_EN.jsonl --batch-size 100 > audiobox_output_WutheringWaves2.0_EN.jsonl
# CUDA_VISIBLE_DEVICES=1 audio-aes audiobox_input_WutheringWaves2.0_CN.jsonl --batch-size 100 > audiobox_output_WutheringWaves2.0_CN.jsonl
# CUDA_VISIBLE_DEVICES=2 audio-aes audiobox_input_WutheringWaves2.0_KR.jsonl --batch-size 100 > audiobox_output_WutheringWaves2.0_KR.jsonl
# CUDA_VISIBLE_DEVICES=3 audio-aes audiobox_input_WutheringWaves2.0_JP.jsonl --batch-size 100 > audiobox_output_WutheringWaves2.0_JP.jsonl
#
# CUDA_VISIBLE_DEVICES=0 audio-aes audiobox_input_StarRail2.7_EN.jsonl --batch-size 100 > audiobox_output_StarRail2.7_EN.jsonl
# CUDA_VISIBLE_DEVICES=1 audio-aes audiobox_input_StarRail2.7_CN.jsonl --batch-size 100 > audiobox_output_StarRail2.7_CN.jsonl
# CUDA_VISIBLE_DEVICES=2 audio-aes audiobox_input_StarRail2.7_KR.jsonl --batch-size 100 > audiobox_output_StarRail2.7_KR.jsonl
# CUDA_VISIBLE_DEVICES=3 audio-aes audiobox_input_StarRail2.7_JP.jsonl --batch-size 100 > audiobox_output_StarRail2.7_JP.jsonl


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

## (4). Plot distributions of different metric values and save audio samples of extreme values.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torchaudio

metrics = ['CE', 'CU', 'PC', 'PQ']
metric_labels = ['Content Enjoyment', 'Content Usefulness', 'Production Complexity', 'Production Quality']
colors = ['blue', 'orange', 'green', 'red']

datasets = ["LibriTTS", \
            "StarRail2.7_EN", "StarRail2.7_CN", "StarRail2.7_KR", "StarRail2.7_JP", \
            "WutheringWaves2.0_EN", "WutheringWaves2.0_CN", "WutheringWaves2.0_KR", "WutheringWaves2.0_JP", \
            "Genshin5.3_EN", "Genshin5.3_CN", "Genshin5.3_KR"
        ] # "Genshin5.3_JP" <--corrupt files.


hist_save_dir = f"/workspace/figures/audiobox_aesthetics/cli/histogram/"
os.makedirs(hist_save_dir, exist_ok=True)
#
pair_save_dir = f"/workspace/figures/audiobox_aesthetics/cli/pairwise_hist/"
os.makedirs(pair_save_dir, exist_ok=True)
#
xcorr_save_dir = f"/workspace/figures/audiobox_aesthetics/cli/xcorr/"
os.makedirs(xcorr_save_dir, exist_ok=True)
#
samp_save_dir = f"/workspace/figures/audiobox_aesthetics/cli/samples/"
os.makedirs(samp_save_dir, exist_ok=True)



for dataset in datasets:
    print(f"{dataset=}")

    in_path = f"audiobox_input_{dataset}.jsonl"
    out_path = f"audiobox_output_{dataset}.jsonl"
    #
    input = pd.read_json(path_or_buf=in_path, lines=True)
    output = pd.read_json(path_or_buf=out_path, lines=True)
    df = input.join(output)



    ## (5). Plot histograms of different metric values.
    for metric, label, color in zip(metrics, metric_labels, colors):    
        sns.histplot(df[metric], kde=True, label=label, color=color, alpha=0.4)
    plt.legend()
    plt.title(f"Audiobox Aesthetics for {dataset}")
    plt.savefig(f"{hist_save_dir}{dataset}.png", dpi=300)
    plt.close()


    # (6). Plot pairwise histograms/distributions for different metric values.
    sns.pairplot(df[metrics], diag_kind="hist") 
    plt.suptitle(f"Audiobox Aesthetics for {dataset}")
    plt.savefig(f"{pair_save_dir}{dataset}.png", dpi=300)
    plt.close()


    # (7). Cross-correlation for different metrics.
    corr_matrix = df[metrics].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Audiobox Aesthetics for {dataset}")
    plt.savefig(f"{xcorr_save_dir}{dataset}.png", dpi=300)
    plt.close()


    # (8). Save out a couple samples that are high and low in each metric.
    num_samples=5

    for metric in metrics:
        df_sorted = df.sort_values(by=metric, ascending=False)
        for s in range(num_samples):
            hi_wv, hi_sr = torchaudio.load(df_sorted.iloc[s].path)
            lo_wv, lo_sr = torchaudio.load(df_sorted.iloc[-(s+1)].path)
            #
            hi_met = f"{df_sorted.iloc[s][metric]:0.2f}"
            lo_met = f"{df_sorted.iloc[-(s+1)][metric]:0.2f}"
            #
            # save samples of high metric and low metric values to listen to them
            torchaudio.save(f"{samp_save_dir}/{dataset}_{metric}_{hi_met}.wav", hi_wv, hi_sr) 
            torchaudio.save(f"{samp_save_dir}/{dataset}_{metric}_{lo_met}.wav", lo_wv, lo_sr) 


# import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)







