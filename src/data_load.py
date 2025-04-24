import librosa
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor
import torch
import random

CORE_LABELS = {"yes","no","up","down","left","right","on","off","stop","go"}

def load_speech_commands(
    data_dir: str,
    split: str = "train",
    max_files: int = None,
    sr: int = 16000,
    other_label: str = None
):
    random.seed(42)
    root = Path(data_dir)
    if (root/"train"/"train").is_dir():
        base = root/"train"/"train"
    elif (root/"train").is_dir() and (root/"train"/"validation_list.txt").exists():
        base = root/"train"
    else:
        base = root
    audio_base = base/"audio"

    val_set  = set((base/"validation_list.txt").read_text().splitlines())
    test_set = set((base/"testing_list.txt").read_text().splitlines())

    missing = []
    for rel_path in sorted(val_set | test_set):
        if not (audio_base/rel_path).exists():
            missing.append(rel_path)
    if missing:
        print("Brakujące pliki:")
        for p in missing:
            print(f"  {p}")

    def rel(p: Path):
        return p.relative_to(audio_base).as_posix()

    if split in ("train", "validation"):
        all_wavs = sorted(audio_base.glob("*/*.wav"))
        if split == "train":
            wavs = [p for p in all_wavs
                    if rel(p) not in val_set and rel(p) not in test_set]
        else:  
            wavs = [p for p in all_wavs if rel(p) in val_set]

    elif split == "test":
        test_audio = root/"test"/"test"/"audio"
        wavs = sorted(test_audio.glob("*.wav"))
    else:
        raise ValueError(f"Nieznany split: {split}")
    
    random.shuffle(wavs)
    
    if max_files is not None:
        wavs = wavs[:max_files]

    dataset = []
    for p in wavs:
        wav, _ = librosa.load(p, sr=sr)
        if len(wav) < sr:
            wav = np.pad(wav, (0, sr - len(wav)), mode="constant")
        else:
            wav = wav[:sr]
        orig_label = p.parent.name if split!="test" else p.stem

        if other_label and orig_label not in CORE_LABELS:
            label = other_label
        else:
            label = orig_label

        dataset.append((wav, label))

    return dataset


class SpeechCommandsDataset(Dataset):
    def __init__(self, data, label2id, sr=16000):
        self.data = data 
        self.label2id = label2id
        self.sr = sr
        self.fe = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-speech-commands-v2")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav, lbl = self.data[idx]
        inputs = self.fe(
            wav,
            sampling_rate=self.sr,
            return_tensors="pt"
        )
        spec = inputs["input_values"].squeeze(0)
        label = torch.tensor(self.label2id[lbl], dtype=torch.long)
        return spec, label