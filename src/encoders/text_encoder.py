import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from src.utils.read_config import config


class BGEEncoder:
    def __init__(self, model_path=config.get("model_path").get('bge_encode')):
        """编码器"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.float16,  # 半精度加速
        ).to(self.device)

        self.model.eval()

    def encode(self, texts: list, batch_size: int = 128, normalize:bool=False) -> np.ndarray:
        """编码字符串列表为向量"""
        if not texts:
            return np.array([])

        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )

                # 移动到GPU
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 编码
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]

                if normalize:
                    # 归一化
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                # 收集结果
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings) if all_embeddings else np.array([])


# 使用示例
if __name__ == "__main__":
    # 你的代码
    sentences = [ "我喜欢睡觉"]
    model_path = r"E:\agent-env\model\embedding\BAAI--bge-large-zh-v1.5"

    # 编码
    encoder = BGEEncoder(model_path)
    embeddings = encoder.encode(sentences, batch_size=128)

    # 输出结果
    print(f"向量形状: {embeddings.shape}")
    # print(f"第一个向量: {embeddings[0][:10]}...")