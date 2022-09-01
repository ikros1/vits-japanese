import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import IPython.display as ipd

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence, cleaned_text_to_sequence
from text.cleaners import japanese_cleaners
from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    # print(text_norm.shape)
    return text_norm


hps_ms = utils.get_hparams_from_file("configs/japanese_base.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model).cuda()


def jtts(spkid, text):
    sid = torch.LongTensor([spkid])  # speaker identity
    stn_tst = get_text(text, hps_ms)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        # print(stn_tst.size())
        audio = net_g_ms.infer(x_tst.cuda(), x_tst_lengths.cuda(), sid=sid.cuda(), noise_scale=.667, noise_scale_w=0.8,
                               length_scale=1)[0][
            0, 0].data.float().cpu().numpy()
    ipd.display(ipd.Audio(audio, rate=hps_ms.data.sampling_rate))


_ = utils.load_checkpoint("logs/output.pth", net_g_ms, None)

jtts(2, "いやだ、おにじゃんかきらい")
