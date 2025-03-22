import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from torch.profiler import profile, ProfilerActivity
import contextlib

from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm

# 1). I have been running from inside zaudio_preprocessing docker container enviornment.
# 2). Install this before import with "python3 -m uv pip install -e /workspace/audiobox-aesthetics"
from audiobox_aesthetics.infer import make_inference_batch
from audiobox_aesthetics.model.aes import AesMultiOutput, Normalize

## Run This With::
        # seq 0 7 | xargs -P 8 -I {} env GLOBAL_RANK={} WORLD_SIZE=8 LRM=8 python3 audiobox-aesthetics/Zyphra_Aes/compute_Aes.py


AXES_NAME = ["CE", "CU", "PC", "PQ"]
metric_labels = ['Content Enjoyment', 'Content Usefulness', 'Production Complexity', 'Production Quality']
colors = ['blue', 'orange', 'green', 'red']

GLOBAL_RANK = int(os.getenv("GLOBAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

SAMPLE_RATE = 16000 # constant for audiobox-aesthetics

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
        # sr = self.sr_list[idx]
        # num_samples = self.num_samples_list[idx]
        # window_snips = self.index_map[str(afile)]

        wav, sr = torchaudio.load(afile)

        # Think that Aes only takes in 16kHz sample rate.
        wav = torchaudio.functional.resample(
            wav,
            orig_freq=sr,
            new_freq=SAMPLE_RATE,
        )

        # convert stereo to mono
        if wav.shape[0] == 2:
            wav = wav.mean(0, keepdims=True)

        # Compute splits based on duration of audio sample and put into tuple to use as key with filename
        snip_times = []
        samples_per_window = window_size*SAMPLE_RATE  # samples in window
        samples_per_hop = hop_size*SAMPLE_RATE        # samples in hop
        total_samples = wav.shape[1]

        # Create multiple segments for long audio or a single segment if short.
        for start in range(0, total_samples, samples_per_hop):
            snip_times.append( (start, min(start+samples_per_window-1,total_samples) ) )

        wavs, masks, weights, bids = make_inference_batch(
            input_wavs=[wav],
            hop_size=self.hop_size,
            window_size=self.window_size,
            sample_rate=SAMPLE_RATE,
            pad_zero=True,
        )

        wavs = torch.stack(wavs)
        masks = torch.stack(masks)
        weights = torch.tensor(weights)
        bids = torch.tensor(bids)
        srs = torch.full((wavs.shape[0],), SAMPLE_RATE)

        return {
            "wavs": wavs,
            "masks": masks,
            "weights": weights,
            "bids": bids,
            "srs": srs,
            "filename": afile,
            "window_snips": snip_times,
        }

def custom_collate(batch):
    wavs = torch.cat([item["wavs"] for item in batch], dim=0)
    masks = torch.cat([item["masks"] for item in batch], dim=0)
    weights = torch.cat([item["weights"] for item in batch], dim=0)
    bids = torch.cat([item["bids"] for item in batch], dim=0)
    srs = torch.cat([item["srs"] for item in batch], dim=0)
    keys = [f'{item["filename"]}_{start}_{end}' for item in batch for (start,end) in item["window_snips"]]

    return wavs, masks, weights, bids, srs, keys




if __name__ == "__main__":

    # dataset_name='twitch'
    # basedir = Path("/datasets/raw/twitch/")
    # filetype = ".mp3"
    # bs = (100,1)

    # dataset_name = 'LibriTTS'
    # basedir = Path("/datasets/raw/LibriTTS-R/")
    # filetype = ".wav"
    # bs = (256,512)

    # dataset_name = 'expresso'
    # basedir = Path("/datasets/raw/expresso/")
    # filetype = ".wav"
    # bs = (100,16)

    # dataset_name = 'EARS'
    # basedir = Path("/datasets/raw/EARS/")
    # filetype = ".wav"
    # bs = (100,16)

    # dataset_name = 'Genshin_EN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/Genshin5.3/Genshin5.3_EN/")
    # filetype = ".wav"
    # bs = (64,32)

    # dataset_name = 'StarRail_EN' # BROKEN FOR NOW?? CORRUPT DATA??
    # basedir = Path("/datasets/raw/games_updated/gaming1/StarRail2.7/StarRail2.7_EN/")
    # filetype = ".wav"
    # bs = (128,16)

    dataset_name = 'WuWaves_EN'
    basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_EN/")
    filetype = ".wav"
    bs = (128,64)

    # dataset_name = 'WuWaves_CN'
    # basedir = Path("/datasets/raw/games_updated/gaming1/WutheringWaves2.0/WutheringWaves2.0_CN/")
    # filetype = ".wav"
    # bs = (256,16)

    audio_files = list(basedir.glob(f"**/*{filetype}"))
    print(f"{len(audio_files)=}")

    # NOTE:  Warning: Xing stream size off by more than 1%, fuzzy seeking may be even more fuzzy than by design!
    #       This warning means that mp3 file is corrupted somehow and should be re-encoded with FFMPEG
    #       because the fine timing is not reliable and alignment of audio will not be correct.
    #

    device = f"cuda:{GLOBAL_RANK}"      # NOTE: tune batch_sizes while watching "nvtop -d 0.5"
    batch_size_GPU = bs[0]              # number of wavs to feed into GPU at once (tune based on window_size when chopping long audio)
    batch_size_DL = bs[1]               # number of audio files processed in DataLoader at once. (make larger for short files)
    hop_size = 10                       # seconds
    window_size = 10                    # seconds

    output_dir = 'figures/audiobox_aesthetics/Aes/'
    os.makedirs(output_dir, exist_ok=True)

    USE_TORCH_PROFILER = False

    dataset = AudioDataset(audio_files, window_size, hop_size)

    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True)

    # print(f"In main after dataset construction")
    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size_DL,
        num_workers=2, 
        collate_fn=custom_collate,
        sampler=sampler,
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
    model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics").to(device)
    target_transform = {
        axis: Normalize(
            mean=model.target_transform[axis]["mean"],
            std=model.target_transform[axis]["std"],
        )
        for axis in AXES_NAME
    }
    #
    model.eval()
    with torch.no_grad(), prof:
        for wavs, masks, weights, bids, srs, keys in tqdm(dataloader, desc="looping over dataloader"):
            cnt += 1

            # if not (wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0] == srs.shape[0] == len(keys)):
            #     import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

            # if wavs is None:
            #     print('wavs was None, prob because of multiple sample rates and mismatched vector sizes in torch.cat in collate. Skip')
            #     continue


            # if cnt<4090:
            #     print('skip ahead')
            #     continue

            

            keys_list.extend(keys)
            for start in range(0, wavs.size(0), batch_size_GPU):
                end = start + batch_size_GPU
                wav_batch = wavs[start:end].to(device, non_blocking=True)
                mask_batch = masks[start:end].to(device, non_blocking=True)
                #
                output = model({"wav": wav_batch, "mask": mask_batch})
                #
                out_list['CE'].append(output['CE'].to('cpu', non_blocking=True))
                out_list['CU'].append(output['CU'].to('cpu', non_blocking=True))
                out_list['PC'].append(output['PC'].to('cpu', non_blocking=True))
                out_list['PQ'].append(output['PQ'].to('cpu', non_blocking=True))

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
    
                

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #
    # Back to CPU here ...
    #
    # convert list of tensors into long tensors

    preds_all = {'CE': torch.cat(out_list['CE'],dim=0),
                 'CU': torch.cat(out_list['CU'],dim=0),
                 'PC': torch.cat(out_list['PC'],dim=0),
                 'PQ': torch.cat(out_list['PQ'],dim=0)
    }

    # Gotta do some Normalizing and Junk Here.. on CPU.
    # The other junk was Weighted averaging, which I did away with.
    # Think this is doing the same thing as basically, the important thing.
    # https://github.com/facebookresearch/audiobox-aesthetics/blob/main/src/audiobox_aesthetics/infer.py#L187-L202
    all_result = {}
    for axis in AXES_NAME:
        preds = target_transform[axis].inverse(preds_all[axis])
        all_result[axis] = preds # weighted_preds

    ## Pack keys and all results into dataframe and save it as csv / json.
    print(f"Saving data to tsv.")
    data = {
        'key': keys_list,
        'CE': all_result['CE'].tolist(),
        'CU': all_result['CU'].tolist(),
        'PC': all_result['PC'].tolist(),
        'PQ': all_result['PQ'].tolist(),
    }

    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

    df = pd.DataFrame(data)

    tsv_save_dir = output_dir + f'data_tsv/{dataset_name}/'
    os.makedirs(tsv_save_dir, exist_ok=True)
    df.to_csv(f"{tsv_save_dir}{dataset_name}_{dataloader.sampler.rank}.tsv", index=False, sep='\t')






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
                                               num_samples=end_hi - start_hi)
                lo_wv, lo_sr = torchaudio.load(fname_lo,
                                               frame_offset=start_lo,
                                               num_samples=end_lo - start_lo)
                #
                hi_met = f"{df_sorted.iloc[s][metric]:0.2f}"
                lo_met = f"{df_sorted.iloc[-(s+1)][metric]:0.2f}"
                #
                # save samples of high metric and low metric values to listen to them
                torchaudio.save(f"{samp_save_dir}{metric}_{hi_met}_{fname_hi.split('/')[-1].removesuffix(filetype)}_{str(start_hi)}_{str(start_hi)}{filetype}", hi_wv, hi_sr) 
                torchaudio.save(f"{samp_save_dir}{metric}_{lo_met}_{fname_lo.split('/')[-1].removesuffix(filetype)}_{str(start_lo)}_{str(start_lo)}{filetype}", lo_wv, lo_sr) 

    # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)

