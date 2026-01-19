
from ...utils.signal import nanopore_process_signal
from .process_data import sliding_window_chunks,process_read
import faiss
import gzip
import json
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
from abc import ABC, abstractmethod
import os

# 基类：抽象类
class InterfaceTokenizer(ABC):
    @abstractmethod
    def tokenize_data(self, signal: np.ndarray) -> list:
        """将原始信号数据转换为 token 字符串"""
        pass

    @abstractmethod
    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        """将测序读段（read）对象转换为 token 字符串"""
        pass

    @abstractmethod
    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        """从 FAST5 文件中读取信号并保存 token 到输出路径"""
        pass

class KmeansTokenizer(InterfaceTokenizer):
    """
    Nanopore RVQ Tokenizer 封装类。

    功能：
        - 加载预训练 RVQ 模型
        - tokenize 单个 read / numpy 信号 / 整个 FAST5 目录
    """

    def __init__(
        self,
        centroids_path: str,
    ):
        """
        初始化 tokenizer。
        """
        data = np.load(centroids_path, allow_pickle=True).item()
        self.window_size = data["dimension"]
        self.stride = data["stride"]
        self.index = self._init_worker(data["centroids"])

    def _init_worker(self, centroids):
        d = centroids.shape[1]
        if hasattr(faiss, 'StandardGpuResources'):
        # === GPU 模式 ===
            print("🚀 Initializing FAISS GPU index...")
            res = faiss.StandardGpuResources()  # GPU 资源管理器
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(centroids) # type: ignore
            # 将 CPU 索引搬到 GPU（默认 device=0）
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            # === CPU 回退模式 ===
            print("💻 Using FAISS CPU index...")
            cpu_index = faiss.IndexFlatL2(d)
            cpu_index.add(centroids) # type: ignore
            index = cpu_index
        return index
    

    def tokenize_data(self, signal: np.ndarray) -> list:
        if signal.size == 0:
            return []
        vec_list = sliding_window_chunks(signal, self.window_size, self.stride)
        if not vec_list:
            return []
        try:
            X = np.stack(vec_list, axis=0).astype(np.float32)
        except Exception:
            return []
        _, I = self.index.search(X, 1) # type: ignore
        cluster_ids = I[:, 0].tolist()

        parts = []
        for token_id in cluster_ids:
            parts.append(f"<|bwav:{int(token_id)}|>")
        return parts


    def tokenize_read(self, read, nanopore_signal_process_strategy="apple") -> list:
        signal_raw = process_read(read)
        if signal_raw is None:
            return []

        signal_processed = nanopore_process_signal(signal_raw,nanopore_signal_process_strategy)
        if signal_processed is None:
            return []
        return self.tokenize_data(signal_processed)


 
    def tokenize_fast5(self, fast5_path: str, output_path:str, nanopore_signal_process_strategy="apple"):
        print(f"✅ Processing {fast5_path} with strategy{nanopore_signal_process_strategy}")
        results = []
        with get_fast5_file(fast5_path, mode="r") as f5:
            for read in tqdm(f5.get_reads(), desc=os.path.basename(fast5_path)):
                try:
                    token_list = self.tokenize_read(read,nanopore_signal_process_strategy)
                    token_str = "".join(token_list)
                    results.append({"id": read.read_id, "text": token_str})
                except Exception as e:
                    print(f"❌ Failed on read {read.read_id}: {e}")
                    continue

        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        print(f"✅ Wrote {len(results)} reads to {output_path}")