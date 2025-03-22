from audiobox_aesthetics.infer import make_inference_batch # AesPredictor, initialize_predictor, 
from audiobox_aesthetics.model.aes import AesMultiOutput

import torch
import torchaudio
from pathlib import Path
from line_profiler import profile

@profile
def load_and_batch_wav(fname: str | Path , 
                       window_size: int = 10, 
                       hop_size: int = 10,
                       device = "cuda:0"):

    wav, sr = torchaudio.load(afile)

    # convert wav to mono if stereo
    if wav.shape[0]==2: 
        wav = wav.mean(0, keepdims=True)

    wavs, masks, weights, bids = make_inference_batch(
                                        input_wavs = [wav],
                                        hop_size=hop_size,
                                        window_size=window_size,
                                        sample_rate=sr,
                                        pad_zero=True,
    )

    # collate
    wavs = torch.stack(wavs)#.to(device)
    masks = torch.stack(masks)#.to(device)
    weights = torch.tensor(weights)#.to(device)
    bids = torch.tensor(bids)#.to(device)

    assert wavs.shape[0] == masks.shape[0] == weights.shape[0] == bids.shape[0]

    return wavs, masks, weights, bids, sr


@profile
def feed_thru_model(model, 
                    wavs, 
                    mask, 
                    batch_size):



    # do this in a for loop.
    xxx = model({"wav": wavs[:batch_size].to(device), 
                 "mask": masks[:batch_size].to(device)}
    )


    return xxx



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":

    basedir = Path("/datasets/raw/twitch/")

    audio_files = list(basedir.glob("*.mp3"))

    device = "cuda:0"

    bs = 100

    model = AesMultiOutput.from_pretrained("facebook/audiobox-aesthetics").to(device)

    for afile in audio_files:

        wavs, masks, weights, bids, sr = load_and_batch_wav(afile)

        xxx = feed_thru_model(model, wavs, masks, bs)

        # import IPython; print('\n\n\Debug:'); IPython.embed(); import time;  time.sleep(0.3)




    