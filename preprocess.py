import os
import librosa
import soundfile as sf
import noisereduce as nr
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def preprocess_file(file_tuple, target_sr=16000, trim_silence=True, reduce_noise=False):
    in_path, out_path = file_tuple
    try:
        y, sr = librosa.load(in_path, sr=None, mono=True)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        y = y / max(abs(y)) if max(abs(y)) > 0 else y

        if trim_silence:
            y, _ = librosa.effects.trim(y, top_db=20)

        if reduce_noise:
            y = nr.reduce_noise(y=y, sr=target_sr)

        sf.write(out_path, y, target_sr)
        return True
    except Exception as e:
        print(f"Failed {in_path}: {e}")
        return False

def process_all(input_dir, output_dir, ext=".mp3", target_sr=16000, trim_silence=True, reduce_noise=False):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith(ext)]

    file_paths = [(os.path.join(input_dir, f), os.path.join(output_dir, f)) for f in files]

    with Pool(cpu_count()) as pool:
        worker = partial(
            preprocess_file,
            target_sr=target_sr,
            trim_silence=trim_silence,
            reduce_noise=reduce_noise
        )
        list(tqdm(pool.imap(worker, file_paths), total=len(file_paths), desc="Preprocessing"))

if __name__ == "__main__":
    process_all(
        input_dir="data/raw",
        output_dir="clean_audio",
        ext=".mp3",       
        target_sr=16000,
        trim_silence=True,
        reduce_noise=False  
    )
