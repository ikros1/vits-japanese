# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

### Jaehyeon Kim, Jungil Kong, and Juhee Son

In our recent [paper](https://arxiv.org/abs/2106.06103), we propose VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech.

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

Visit our [demo](https://jaywalnut310.github.io/vits-demo/index.html) for audio samples.

We also provide the [pretrained models](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2?usp=sharing).

** Update note: Thanks to [Rishikesh (ऋषिकेश)](https://github.com/jaywalnut310/vits/issues/1), our interactive TTS demo is now available on [Colab Notebook](https://colab.research.google.com/drive/1CO61pZizDj7en71NQG_aqqKdGaA_SaBf?usp=sharing).

<table style="width:100%">
  <tr>
    <th>VITS at training</th>
    <th>VITS at inference</th>
  </tr>
  <tr>
    <td><img src="resources/fig_1a.png" alt="VITS at training" height="400"></td>
    <td><img src="resources/fig_1b.png" alt="VITS at inference" height="400"></td>
  </tr>
</table>


## Pre-requisites
操作系统 Centos7
我的python版本 3.8.5
补充，官方没提前说的一个硬性要求，由于torch 1.6的版本，显卡算力要求没查到 估计要5.0 5.2左右，算力查询可以参考：https://blog.csdn.net/zyb418/article/details/87972608
按5.2算 显卡就需要是GTX 965M及以上，算力不够的话，就开摆吧（或者可以尝试低版本torch，嘛，pass）
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. `pip install -r requirements.txt` 由于我本地环境复杂，部分库安装失败，暂时继续
    1. You may need to install espeak first: `apt-get install espeak`
0. Download datasets
    1. Download and extract the LJ Speech dataset, then rename or create a link to the dataset folder: `ln -s /path/to/LJSpeech-1.1/wavs DUMMY1` 创建数据集软链接
    1. For mult-speaker setting, download and extract the VCTK dataset, and downsample wav files to 22050 Hz. Then rename or create a link to the dataset folder: `ln -s /path/to/VCTK-Corpus/downsampled_wavs DUMMY2`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
# 运行时缺库补装 No module named 'Cython'
python setup.py build_ext --inplace
# 报错1 error: could not create 'monotonic_align/core.cpython-38-x86_64-linux-gnu.so': No such file or directory ，莫名其妙了

# 官方例子传入了train val test，我们就根据大佬的模板传入个train和val，val为train中截取一部分。
# train里的文件路径用我们上面生成的软链接DUMMY1
# 运行又缺少了pyopenjtalk、janome，补装
# 这个pyopenjtalk异常难装，自求多福吧www
# 为您自己的数据集进行预处理（g2p）。已经提供了用于LJ语音和VCTK的预处理音素。 预处理主要将日文转成了罗马音的cleaned文件，手动转跳过
# Preprocessing (g2p) for your own datasets. Preprocessed phonemes for LJ Speech and VCTK have been already provided.
# python preprocess.py --text_index 1 --filelists filelists/ljs_audio_text_train_filelist.txt filelists/ljs_audio_text_val_filelist.txt filelists/ljs_audio_text_test_filelist.txt 
# python preprocess.py --text_index 2 --filelists filelists/vctk_audio_sid_text_train_filelist.txt filelists/vctk_audio_sid_text_val_filelist.txt filelists/vctk_audio_sid_text_test_filelist.txt

#preprocess.py里面默认的英文，命令我们改成日文 追加命令 --text_cleaners japanese_phrase_cleaners
python preprocess.py --text_index 1 --filelists filelists/ikaros/train.txt filelists/ikaros/val.txt --text_cleaners japanese_phrase_cleaners
# 运行后会从github下载一个包文件，得加个速
```


## Training Exmaple
```sh
# LJ Speech
python train.py -c configs/ljs_base.json -m ljs_base

# VCTK
python train_ms.py -c configs/vctk_base.json -m vctk_base

# 我们的配置文件就参考大佬提供的新的nan.json，注意nan.json里面的配置项
python train.py -c configs/nan.json -m nan
# 报错1 OSError: cannot load library '..\site-packages\_soundfile_data\libsndfile64bit.dll': error 0x7e
# 缺失文件啊，手动创建文件夹，补充dll文件，下载地址：https://github.com/bastibe/libsndfile-binaries
# 报错2 AssertionError: CPU training is not allowed. 显而易见 没GPU跑不了，具体torch版本对应的显卡算力是有要求的，可以参考 https://blog.csdn.net/qq_43391414/article/details/110562749
# 报错3（linux） If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0
# 补装pip3 install protobuf==3.19.0
# 报错4 ModuleNotFoundError: No module named '_lzma'  ，参考：https://zhuanlan.zhihu.com/p/404162713

```


## Inference Example
See [inference.ipynb](inference.ipynb)
