import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.profiler import profile, ProfilerActivity
import contextlib

from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

# 1). I have been running from inside zaudio_preprocessing docker container enviornment.
# 2). Install this before import with "python3 -m uv pip install -e /workspace/audiobox-aesthetics"
from audiobox_aesthetics.infer import make_inference_batch
from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize

from audiobox_aesthetics.infer import initialize_predictor


## Run This With::
        # seq 0 7 | xargs -P 8 -I {} env GLOBAL_RANK={} WORLD_SIZE=8 LRM=8 python3 audiobox-aesthetics/Zyphra_Aes/compute_Aes.py


AXES_NAME = ["CE", "CU", "PC", "PQ"]
metric_labels = ['Content Enjoyment', 'Content Usefulness', 'Production Complexity', 'Production Quality']
colors = ['blue', 'orange', 'green', 'red']

GLOBAL_RANK = int(os.getenv("GLOBAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

SAMPLE_RATE=16000

def compute_sample_snips(wav, sr, window_size, hop_size):
    snip_times = []
    total_samples = wav.shape[1]
    samples_per_hop = hop_size*sr
    samples_per_window = window_size*sr

    # Create multiple segments for long audio or a single segment if short.
    for start in range(0, total_samples, samples_per_hop):
        snip_times.append( (start, min(start+samples_per_window-1,total_samples) ) )

    return snip_times

class AudioDataset(Dataset):
    def __init__(self, file_list, window_size=10, hop_size=10):
        
        print(f"Inside AudioDataset __init__,")
        self.file_list = file_list
        self.window_size = window_size
        self.hop_size = hop_size

    def __len__(self):
        # return number of samples
        return len(self.file_list)

    def __getitem__(self, idx):
        afile = self.file_list[idx]
        
        wav, sr = torchaudio.load(afile)

        # compute sample start and ends in original sample rate waveform
        snip_times_origSR = compute_sample_snips(wav, sr, self.window_size, self.hop_size)

        # Think that Aes only takes in 16kHz sample rate.
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=SAMPLE_RATE,
        )

        # convert stereo to mono
        if wav.shape[0] == 2:
            wav = wav.mean(0, keepdims=True)

        # compute sample start and ends in 16kHz waveform
        snip_times_16k = compute_sample_snips(wav, SAMPLE_RATE, self.window_size, self.hop_size)

        assert len(snip_times_16k)==len(snip_times_origSR) # if not true, something broke.

        # chop the 16kHz waveform into a list of tensors based on snip_times_16k
        num_snips = len(snip_times_16k)
        wavs = []
        for snip_start,snip_end in snip_times_16k:
            wavs.append(wav[:,snip_start:snip_end])

        return {
            "wavs": wavs,
            "srs_16k": [SAMPLE_RATE] * num_snips,
            "srs_orig": [sr] * num_snips,
            "snip_times_16k": snip_times_16k,
            "snip_times_orig": snip_times_origSR,
            "fnames": [str(afile)] * num_snips,
        }

def custom_collate(batch):

    wavs = [wav for item in batch for wav in item["wavs"]]
    srs = [sr for item in batch for sr in item["srs_16k"]]
    fnames = [fname for item in batch for fname in item["fnames"]]
    snips_16k = [snips for item in batch for snips in item["snip_times_16k"]]
    snips_orig = [snips for item in batch for snips in item["snip_times_orig"]]

    output = []
    for i in range(len(wavs)):
        output.append(
            {"path" : wavs[i],
             "sample_rate" : srs[i],
             "fname" : fnames[i],
             "snips_16k" : snips_16k[i],
             "snips_orig" : snips_orig[i],
            }
        )

    return output



if __name__ == "__main__":

    # dataset_name='twitch'
    # basedir = Path("/datasets/raw/twitch/")               # FILES TOO BIG. OOMS GPU. IMPLEMENT FIX.
    # filetype = ".mp3"
    # bs = (1,256)

    # dataset_name = 'emilia'
    # basedir = Path("/datasets/raw/emilia_separated/")     # FILES TOO BIG. OOMS GPU. IMPLEMENT FIX.
    # filetype = ".opus"
    # bs = (1,256)

    # podcasts                                              # FILES TOO BIG. OOMS GPU. IMPLEMENT FIX.

    # dataset_name = 'VCTK'
    # basedir = Path("/datasets/raw/VCTK-Corpus-0.92/")
    # filetype = ".flac"
    # bs = (256,1024)

    # dataset_name = 'LibriTTS'
    # basedir = Path("/datasets/raw/LibriTTS-R/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'expresso'
    # basedir = Path("/datasets/raw/expresso/")
    # filetype = ".wav"
    # bs = (4,1024)

    # dataset_name = 'EARS'
    # basedir = Path("/datasets/raw/EARS/")
    # filetype = ".wav"
    # bs = (32,1024)

    # dataset_name = 'hfc'
    # basedir = Path("/datasets/raw/hfc/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'hi_fi_tts'
    # basedir = Path("/datasets/raw/hi_fi_tts_v0/")
    # filetype = ".flac"
    # bs = (128,1024)

    # dataset_name = 'Genshin_EN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/Genshin5.3/Genshin5.3_EN/")
    # filetype = ".wav"
    # bs = (80,1024)

    # dataset_name = 'Genshin_CN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/Genshin5.3/Genshin5.3_CN/")
    # filetype = ".wav"
    # bs = (80,1024)

    # dataset_name = 'Genshin_JP'
    # basedir = Path("/datasets/raw/games_updated/gaming1/Genshin5.3/Genshin5.3_JP/") # A BROKEN FILE IN HERE.
    # filetype = ".wav"
    # bs = (80,1024)

    # dataset_name = 'Genshin_KR'
    # basedir = Path("/datasets/raw/games_updated/gaming1/Genshin5.3/Genshin5.3_KR/")
    # filetype = ".wav"
    # bs = (80,1024)

    # dataset_name = 'StarRail_EN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/StarRail2.7/StarRail2.7_EN/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'StarRail_CN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/StarRail2.7/StarRail2.7_CN/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'StarRail_JP'
    # basedir = Path("/datasets/raw/games_updated/gaming1/StarRail2.7/StarRail2.7_JP/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'StarRail_KR'
    # basedir = Path("/datasets/raw/games_updated/gaming1/StarRail2.7/StarRail2.7_KR/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'WuWaves_EN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_EN/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'WuWaves_CN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_CN/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'WuWaves_JP'
    # basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_JP/")
    # filetype = ".wav"
    # bs = (128,1024)

    # dataset_name = 'WuWaves_KR'
    # basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_KR/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'arknights_en'
    # basedir = Path("/datasets/raw/games_updated/arknights/arknights_voices_en/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'arknights_jp'
    # basedir = Path("/datasets/raw/games_updated/arknights/arknights_voices_jp/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'arknights_kr'
    # basedir = Path("/datasets/raw/games_updated/arknights/arknights_voices_kr/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'arknights_zh'
    # basedir = Path("/datasets/raw/games_updated/arknights/arknights_voices_zh/")
    # filetype = ".wav"
    # bs = (256,1024)

    # dataset_name = 'azurlane_jp'
    # basedir = Path("/datasets/raw/games_updated/azurlane_voices_jp/")
    # filetype = ".ogg"
    # bs = (256,1024)

    # dataset_name = 'girlsfrontline_jp'
    # basedir = Path("/datasets/raw/games_updated/girlsfrontline_voices_jp/")
    # filetype = ".ogg"
    # bs = (256,1024)

    # dataset_name = 'moe_speech'
    # basedir = Path("/datasets/raw/games_updated/moe_speech/data/")
    # filetype = ".wav"
    # bs = (256,1024)


    # seedtts_testset   ?
    # mls_dutch         ?
    # mls_english  
    # mls_french  
    # mls_german
    # mls_italian
    # mls_polish
    # mls_portuguese
    # mls_spanish


    # dataset_name = 'daps'
    # basedir = Path("/datasets/raw/daps/")  # A BROKEN FILE IN HERE.
    # filetype = ".wav"
    # bs = (256,1024)



    



    audio_files = list(basedir.glob(f"**/*{filetype}"))
    print(f"{len(audio_files)=}")

    # NOTE:  Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
    #       This warning means that mp3 file is corrupted somehow and should be re-encoded with FFMPEG
    #       because the fine timing is not reliable and alignment of audio will not be correct.
    #

    device = f"cuda:{GLOBAL_RANK}"      # NOTE: tune batch_sizes while watching "nvtop -d 0.5"
    batch_size_DL = bs[0]               # number of audio files processed in DataLoader at once. (make larger for short files)
    batch_size_max_GPU = bs[1]              # number of wavs to feed into GPU at once (tune based on window_size when chopping long audio)
    hop_size = 10                       # seconds
    window_size = 10                    # seconds

    output_dir = 'figures/audiobox_aesthetics/Aes/'
    os.makedirs(output_dir, exist_ok=True)

    USE_TORCH_PROFILER = False

    dataset = AudioDataset(audio_files, window_size, hop_size)

    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_DL,
        num_workers=2, 
        collate_fn=custom_collate,
        # sampler=sampler,
        prefetch_factor=2,
        pin_memory=True,
    )

    cnt = 0

    out_list = {'CE':[], 
                'CU':[], 
                'PC':[], 
                'PQ':[]
    }
    keys_list = []

    # Set USE_PROFLER to True to record an Aes_trace.json file we can inspect in perfetto.
    if USE_TORCH_PROFILER:
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
    else:
        prof = contextlib.nullcontext()




    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # This accesses the GPU.
    #
    if False:
        predictor_16k = initialize_predictor()

        predictor_48k = initialize_predictor()
        predictor_48k.sample_rate=48000

        L16 = {'CE':[], 
            'CU':[], 
            'PC':[], 
            'PQ':[]
        }
        L48 = {'CE':[], 
            'CU':[], 
            'PC':[], 
            'PQ':[]
        }

        for i, audio_file in enumerate(audio_files):
        
            print(i)
            wav, sr = torchaudio.load(audio_file)

            a16 = predictor_16k.forward([{"path": wav, "sample_rate": sr}])
            a48 = predictor_48k.forward([{"path": wav, "sample_rate": sr}])

            for axis in AXES_NAME:
                L16[axis].append(a16[0][axis])
                L48[axis].append(a48[0][axis])


            if i>100:
                break


        import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)


        ## (5). Plot histograms of different metric values.
        df = pd.DataFrame(L48)
        for metric, label, color in zip(AXES_NAME, metric_labels, colors):    
            sns.histplot(df[metric], kde=True, label=label, color=color, alpha=0.4)
        plt.legend()
        plt.title(f"Audiobox Aesthetics for {dataset_name}")
        plt.savefig(f"L48_hist.png", dpi=300)
        plt.close()

        df = pd.DataFrame(L16)
        for metric, label, color in zip(AXES_NAME, metric_labels, colors):    
            sns.histplot(df[metric], kde=True, label=label, color=color, alpha=0.4)
        plt.legend()
        plt.title(f"Audiobox Aesthetics for {dataset_name}")
        plt.savefig(f"L16_hist.png", dpi=300)
        plt.close()

    predictor_16k = initialize_predictor()
    with prof:

        Aes_list = []
        fname_list = []
        starts_16k_list = []
        ends_16k_list = []
        starts_orig_list = []
        ends_orig_list = []

        for batch in tqdm(dataloader, desc="looping over dataloader"):
            cnt += 1

            

            if len(batch) > batch_size_max_GPU:
                print(f"WARNING: BATCH TOO LARGE: {cnt}: {len(batch)=} > {batch_size_max_GPU=}")
                batch = batch[:batch_size_max_GPU]


            out = predictor_16k.forward(batch)
            Aes_list.extend(out)

            fname_list.extend( [item['fname'] for item in batch] )
            starts_16k_list.extend( [item['snips_16k'][0] for item in batch] )
            ends_16k_list.extend( [item['snips_16k'][1] for item in batch] )
            starts_orig_list.extend( [item['snips_orig'][0] for item in batch] )
            ends_orig_list.extend( [item['snips_orig'][1] for item in batch] )

            if USE_TORCH_PROFILER:
                prof.step()

            # print(f"{cnt=}")
            # if cnt >= 10:
            #     if USE_TORCH_PROFILER:  
            #           torch.cuda.synchronize()
            #     break

        if USE_TORCH_PROFILER:
            torch.cuda.synchronize()

    if USE_TORCH_PROFILER:
        prof.export_chrome_trace('Aes_trace.json') # https://ui.perfetto.dev to visualize this with.
        


    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)
                

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # Back to CPU here ...
    #
    CE_list = [Aes['CE'] for Aes in Aes_list]
    CU_list = [Aes['CU'] for Aes in Aes_list]
    PC_list = [Aes['PC'] for Aes in Aes_list]
    PQ_list = [Aes['PQ'] for Aes in Aes_list]


    ## Pack keys and all results into dataframe and save it as csv / json.
    print(f"Saving data to tsv.")
    data = {
        'fname' : fname_list,
        'start_16k' : starts_16k_list,
        'end_16k' : ends_16k_list,
        'start_orig_sr' : starts_orig_list,
        'end_orig_sr' : ends_orig_list,
        'CE': CE_list,
        'CU': CU_list,
        'PC': PC_list,
        'PQ': PQ_list,
    }
    df = pd.DataFrame(data)

    tsv_save_dir = output_dir + f'data_tsv/{dataset_name}/'
    os.makedirs(tsv_save_dir, exist_ok=True)
    df.to_csv(f"{tsv_save_dir}{dataset_name}_{GLOBAL_RANK}.tsv", index=False, sep='\t')






    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # Make Plots from DataFrame. 
    #   NOTE: This should be done in separate script after loading in .tsv file.
    #
    if False:
        print(f"Saving plots.")
        ## (5). Plot histograms of different metric values.
        for metric, label, color in zip(AXES_NAME, metric_labels, colors):    
            sns.histplot(df[metric], kde=True, label=label, color=color, alpha=0.4)
        plt.legend()
        plt.title(f"Audiobox Aesthetics for {dataset_name}")
        plt.savefig(f"{output_dir}{dataset_name}_hist.png", dpi=300)
        plt.close()

        # (6). Plot pairwise histograms/distributions for different metric values.
        sns.pairplot(df[AXES_NAME], diag_kind="hist") 
        plt.suptitle(f"Audiobox Aesthetics for {dataset}")
        plt.savefig(f"{output_dir}{dataset_name}_pairhist.png", dpi=300)
        plt.close()


        # (7). Cross-correlation for different metrics.
        corr_matrix = df[AXES_NAME].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Audiobox Aesthetics for {dataset_name}")
        plt.savefig(f"{output_dir}{dataset_name}_xcorr.png", dpi=300)
        plt.close()

        
        # (8). Save out a couple samples that are high and low in each metric.
        print(f"Saving audio samples.")

        num_samples=32
        for metric in AXES_NAME:
            #
            samp_save_dir = output_dir + f'samples/{metric}/{dataset_name}/'
            os.makedirs(samp_save_dir, exist_ok=True)
            #
            df_sorted = df.sort_values(by=metric, ascending=False)
            for s in range(num_samples):
                fname_hi = df_sorted.iloc[s].key.split(filetype)[0] + filetype
                _, start_hi, end_hi = df_sorted.iloc[s].key.split(filetype)[1].split('_')
                start_hi = int(start_hi)
                end_hi = int(end_hi)
                #
                fname_lo = df_sorted.iloc[-(s+1)].key.split(filetype)[0] + filetype
                _, start_lo, end_lo = df_sorted.iloc[-(s+1)].key.split(filetype)[1].split('_')
                start_lo = int(start_lo)
                end_lo = int(end_lo)                
                #
                hi_wv, hi_sr = torchaudio.load(fname_hi,
                                               frame_offset=start_hi,
                                               num_frames=end_hi - start_hi)
                lo_wv, lo_sr = torchaudio.load(fname_lo,
                                               frame_offset=start_lo,
                                               num_frames=end_lo - start_lo)
                #
                hi_met = f"{df_sorted.iloc[s][metric]:0.2f}"
                lo_met = f"{df_sorted.iloc[-(s+1)][metric]:0.2f}"
                #
                # save samples of high metric and low metric values to listen to them
                torchaudio.save(f"{samp_save_dir}{metric}_{hi_met}_{fname_hi.split('/')[-1].removesuffix(filetype)}_{str(start_hi)}_{str(start_hi)}{filetype}", hi_wv, hi_sr) 
                torchaudio.save(f"{samp_save_dir}{metric}_{lo_met}_{fname_lo.split('/')[-1].removesuffix(filetype)}_{str(start_lo)}_{str(start_lo)}{filetype}", lo_wv, lo_sr) 

    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

