import numpy as np
def sliding_window_chunks(signal, window_size, stride):
    """
    对一维信号进行滑动窗口切片。
    """
    n_points = len(signal)
    if n_points < window_size:
        return []
    chunks_list = []
    start = 0
    while start + window_size <= n_points:
        end = start + window_size
        chunk = signal[start:end]
        chunks_list.append(chunk)
        start += stride
    return chunks_list

def process_read(read):
    signal_raw = None
    try:
        channel_info = read.handle[read.global_key + 'channel_id'].attrs
        offset = int(channel_info['offset'])
        scaling = channel_info['range'] / channel_info['digitisation']
        raw = read.handle[read.raw_dataset_name][:]
        signal_raw = np.array(scaling * (raw + offset), dtype=np.float32)
        return signal_raw
    except Exception as e:
        fast5_path = getattr(read.handle, 'filename', 'unknown.fast5')
        print(f"❌ Error on read {read.read_id} in {fast5_path}: {e}")
        return None