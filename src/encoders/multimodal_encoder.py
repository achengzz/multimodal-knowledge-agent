from src.utils.read_config import config
from typing import Union,List

import numpy as np
from PIL import Image
import torch
from transformers import  AltCLIPModel, AltCLIPProcessor


class AltCLIPEncoder:
    """图片或文本编码"""

    def __init__(self, model_path=config.get("model_path").get('altclip')):
        self.model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AltCLIPModel.from_pretrained(model_path).to(self.model_device)
        self.processor = AltCLIPProcessor.from_pretrained(model_path)
        self.model.eval()
        torch.set_grad_enabled(False)



    def encode_image(self, images:Union[Image.Image, List[Image.Image]],normalize:bool=False)->np.ndarray:
        """
        将一张或多张图片编码
        :param images: 一张或多张图片
        :param normalize: 是否归一化，默认不要。因为在上传到qdrant数据库时，使用cosine会自动归一化
        :return: 编码后的嵌入（一张图片为一个向量，多张图片为矩阵）
        """
        inputs=self.processor(images=images,return_tensors="pt").to(self.model_device)
        with torch.no_grad():
            images_features=self.model.get_image_features(**inputs)

        if normalize:
            images_features=torch.nn.functional.normalize(images_features,p=2,dim=-1)

        embeddings=images_features.squeeze().cpu().numpy()
        return embeddings

    def encode_text(self,texts:Union[str,List[str]],normalize:bool=False)->np.ndarray:
        """
        将一条或多条文本编码成向量
        :param texts: 一条或多条文本
        :param normalize: 是否归一化，默认不要。因为在上传到qdrant数据库时，使用cosine会自动归一化
        :return:
        """
        inputs=self.processor(text=texts,return_tensors="pt",padding=True).to(self.model_device)
        # {'input_ids': tensor([[49406,  1215, 40293, 49407]]), 'attention_mask': tensor([[1, 1, 1, 1]])}
        with torch.no_grad():
            texts_features=self.model.get_text_features(**inputs)

        if normalize:
            texts_features=torch.nn.functional.normalize(texts_features,p=2,dim=-1)

        embeddings=texts_features.squeeze().cpu().numpy()

        return embeddings

    def get_cosine_score(self,vectors_1:np.ndarray,vectors_2:np.ndarray,eps:float=1e-8)->np.ndarray:
        """
        计算两个向量之间的余弦相似度
        :param vectors_1:
        :param vectors_2:
        :return:
        """
        if vectors_1.ndim==1:
            vectors_1=vectors_1.reshape(1,-1)
        if vectors_2.ndim==1:
            vectors_2=vectors_2.reshape(1,-1)

        m,d=vectors_1.shape
        n,_=vectors_2.shape

        dot_matrix=np.dot(vectors_1,vectors_2.T)

        norm_v1=np.linalg.norm(vectors_1,axis=1,keepdims=True)
        norm_v2=np.linalg.norm(vectors_2,axis=1,keepdims=True)

        norm_v1=np.maximum(norm_v1,eps)
        norm_v2=np.maximum(norm_v2,eps)
        similarity_matrix=dot_matrix/(norm_v1 * norm_v2.T)
        similarity_matrix=np.clip(similarity_matrix,-1.0, 1.0)
        return similarity_matrix



if __name__ == '__main__':
    encoder=AltCLIPEncoder()
    images=[Image.open(r"C:\Users\Cheng\Pictures\fish.jpg").convert("RGB"),
            Image.open(r"E:\资料库\舞龙表演.jpg").convert("RGB"),
            Image.open(r"E:\资料库\烤肉聚餐.jpg").convert("RGB"),
            Image.open(r"E:\资料库\2025年前三季度居民人均消费支出及构成.png").convert("RGB"),
            Image.open(r"E:\资料库\2025年前三季度全国及分城乡居民人均可支配收入.png").convert("RGB"),
            Image.open(r"E:\资料库\工业生产者出厂价格涨跌幅.png").convert("RGB"),
            Image.open(r"C:\Users\Cheng\Pictures\框架.png").convert("RGB")]
    images_embeddings=encoder.encode_image(images=images)
    print(images_embeddings.shape)

    texts=[
        "漂亮的金鱼",
        "舞龙表演",
        "烤肉美食",
        "消费支出饼状图",
        "人均收入柱状图",
        "价格波动折线图",
        "软件架构图"
    ]
    texts_embeddings=encoder.encode_text(texts=texts)
    print(texts_embeddings.shape)
    from tabulate import tabulate

    score_matrix=encoder.get_cosine_score(images_embeddings,texts_embeddings)
    labels=["金鱼","舞龙表演","烤肉","支出饼状图","收入柱状图","价格折线图","架构图"]
    print(tabulate(score_matrix, headers=labels, showindex=labels,
                   tablefmt='grid', numalign='center', stralign='center'))
