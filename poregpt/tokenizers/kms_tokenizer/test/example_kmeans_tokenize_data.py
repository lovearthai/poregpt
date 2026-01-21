import numpy as np
from poregpt.tokenizers.kms_tokenizer.kmeans_tokenizer import KmeansTokenizer
from poregpt.utils.signal import nanopore_process_signal
if __name__ == "__main__":
    tokenizer = KmeansTokenizer(
        centroids_path="../models/centroids_meta.npz",
    )
    window_size = tokenizer.window_size
    stride = tokenizer.stride
    # 模拟一段 1200 点的信号（~240ms @ 5kHz）
    signal_raw = np.random.randn(1200).astype(np.float32) * 5 + 100
    signal = nanopore_process_signal(signal_raw, strategy="apple")
    if signal is None:
        print("Signal processing failed.")
        exit(1)
    tokens_all = tokenizer.tokenize_data(signal)
    print(tokens_all)
     