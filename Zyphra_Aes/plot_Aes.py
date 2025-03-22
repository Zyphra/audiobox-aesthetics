import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
import os
import pandas as pd
import torchaudio
import random


AXES_NAME = ["CE", "CU", "PC", "PQ"]
metric_labels = ['Content Enjoyment', 'Content Usefulness', 'Production Complexity', 'Production Quality']
colors = ['blue', 'orange', 'green', 'red']


def save_sample(df, s, metric, audio_type, samp_save_dir):
    """
    Save an audio sample to samp_save_dir given a dataframe (df) output from compute_Aes(_*).py
    and an iloc index s.
    """

    fname= df.iloc[s].fname
    samp_start = df.iloc[s].start_orig_sr
    samp_end = df.iloc[s].end_orig_sr
    #
    wv, sr = torchaudio.load(fname,
                        frame_offset=samp_start,
                        num_frames=samp_end - samp_start)
    #
    met_val = f"{df.iloc[s][metric]:0.2f}"
    #
    torchaudio.save(f"{samp_save_dir}/{metric}_{met_val}_{fname.split('/')[-1].removesuffix(audio_type)}_{str(samp_start)}_{str(samp_end)}{audio_type}", wv, sr) 



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__":

    # dataset_name, audio_type = 'LibriTTS', '.wav'
    # dataset_name, audio_type = 'expresso', '.wav'
    # dataset_name, audio_type = 'EARS', '.wav'
    # dataset_name, audio_type = 'VCTK', '.flac'
    # dataset_name, audio_type = 'hfc', '.wav'
    # dataset_name, audio_type = 'Genshin_EN', '.wav'
    # dataset_name, audio_type = 'Genshin_CN', '.wav'
    # dataset_name, audio_type = 'Genshin_JP', '.wav'         # BROKEN FILE OR SOMETHING.
    # dataset_name, audio_type = 'Genshin_KR', '.wav'
    # dataset_name, audio_type = 'StarRail_EN', '.wav'
    # dataset_name, audio_type = 'StarRail_CN', '.wav'
    # dataset_name, audio_type = 'StarRail_JP', '.wav'
    # dataset_name, audio_type = 'StarRail_KR', '.wav'
    # dataset_name, audio_type = 'WuWaves_EN', '.wav'
    # dataset_name, audio_type = 'WuWaves_CN', '.wav'
    # dataset_name, audio_type = 'WuWaves_JP', '.wav'
    # dataset_name, audio_type = 'WuWaves_KR', '.wav'
    # dataset_name, audio_type = 'arknights_en', '.wav'
    # dataset_name, audio_type = 'arknights_jp', '.wav'
    # dataset_name, audio_type = 'arknights_kr', '.wav'
    # dataset_name, audio_type = 'arknights_zh', '.wav'
    # dataset_name, audio_type = 'azurlane_jp', '.ogg'
    # dataset_name, audio_type = 'girlsfrontline_jp', '.ogg'
    dataset_name, audio_type = 'moe_speech', '.wav'

    num_samples=32

    base_dir = Path("figures/audiobox_aesthetics/Aes/")
    tsv_files_dir = base_dir / f"data_tsv/{dataset_name}/"
    tsv_files = list(tsv_files_dir.glob(f"**/*{dataset_name}*.tsv"))
    #
    plots_dir = base_dir / f"plots/{dataset_name}/"
    os.makedirs(plots_dir, exist_ok=True)

    # Load in all the tsv files pertaining to dataset and concatenate them into 1 big one.
    df = pd.DataFrame()
    for tsv_file in tsv_files:
        df2 = pd.read_csv(tsv_file, sep='\t')
        df = pd.concat([df,df2],ignore_index=True)
        print(f"{tsv_file} -> {len(df)}")


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # Make Plots from DataFrame. 
    #
    print(f"Saving plots.")
    ## (5). Plot histograms of different metric values.
    for metric, label, color in zip(AXES_NAME, metric_labels, colors):    
        sns.histplot(df[metric], kde=True, label=label, color=color, alpha=0.4)
    plt.legend()
    plt.title(f"Audiobox Aesthetics for {dataset_name}")
    plt.savefig(f"{plots_dir}/{dataset_name}_hist.png", dpi=300)
    plt.close()

    ## (6). Plot CDF of different metric values.
    for metric, label, color in zip(AXES_NAME, metric_labels, colors):    
        sns.ecdfplot(df[metric], label=label, color=color, alpha=1)
    plt.legend()
    plt.grid()
    plt.title(f"Audiobox Aesthetics for {dataset_name} - {len(df)} samples.")
    plt.savefig(f"{plots_dir}/{dataset_name}_CDF.png", dpi=300)
    plt.close()

    # (7). Plot pairwise histograms/distributions for different metric values.
    sns.pairplot(df[AXES_NAME], diag_kind="hist") 
    plt.suptitle(f"Audiobox Aesthetics for {dataset_name}")
    plt.savefig(f"{plots_dir}/{dataset_name}_pairhist.png", dpi=300)
    plt.close()


    # (8). Cross-correlation for different metrics.
    corr_matrix = df[AXES_NAME].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Audiobox Aesthetics for {dataset_name}")
    plt.savefig(f"{plots_dir}/{dataset_name}_xcorr.png", dpi=300)
    plt.close()

    
    # (9). #10 BELOW IS BETTER: Save out a couple samples that are high and low in each metric.
    if False:
        print(f"Saving audio samples.")
        for metric in AXES_NAME:
            #
            samp_save_dir = base_dir / f'samples/{metric}/{dataset_name}/'
            os.makedirs(samp_save_dir, exist_ok=True)
            #
            df_sorted = df.sort_values(by=metric, ascending=False)
            for s in range(num_samples):
                save_sample(df_sorted, s, metric, audio_type, samp_save_dir) # save samples where metric is high
                save_sample(df_sorted, -(s+1), metric, audio_type, samp_save_dir) # save samples where metric is low



    # (10). Save samples for intervals of each metric.
    print(f"Saving {num_samples} audio samples for each interval.")
    intervals = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10)]
    for metric in AXES_NAME:
        for interval in intervals:
            filtered_df = df[(df[metric] >= interval[0]) & (df[metric] < interval[1])]
            num_in_interval = len(filtered_df)
            if num_samples > num_in_interval:
                samples = list(range(num_in_interval))
            else:
                samples = random.sample(range(num_in_interval), num_samples)

            samp_save_dir = base_dir / f'samples/{metric}/{dataset_name}/{metric}_{interval[0]}_{interval[1]}_n{num_in_interval}'
            os.makedirs(samp_save_dir, exist_ok=True)

            # print(f"{metric} in {interval} -> {num_in_interval} samples.")
            for s in samples:
                save_sample(filtered_df, s, metric, audio_type, samp_save_dir) # save samples where metric is high


