#多模态向量管理，存储只需传入前好的向量，直接调用即可上传至qdrant
import uuid
from typing import Union, Optional

import numpy as np
from qdrant_client import QdrantClient,models


class VectorStoreManager:
    def __init__(self,location):
        self.client = self._connect(location)

    def _connect(self,location):
        try:
            client = QdrantClient(location)
            return client
        except Exception as e:
            raise ConnectionError(f"无法连接的Qdrant（{location}）:{e}")

    def get_all_collections(self):
        collections=self.client.get_collections()
        collections=collections.collections
        return [collection.name for collection in collections]

    def get_collection_info(self,collection_name,p=False):
        """检查集合信息"""
        collection_info = self.client.get_collection(collection_name)
        if p:
            print(f"集合 '{collection_name}' 信息:")
            print(f"  向量数量: {collection_info.points_count}")
            print(f"  向量维度: {collection_info.config.params.vectors.size}")
            print(f"  配置: {collection_info.config}")
        return collection_info

    def create_collection(self,collection_name,embeddings_size,metadata=None):
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embeddings_size,
                    distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(full_scan_threshold=1000),
                metadata=metadata
            )
            print(f"{collection_name}向量集创建成功")

        except Exception as e:
            raise Exception(f"{collection_name}创建失败！！！{e}")

    def collection_exist(self,collection_name):

        return self.client.collection_exists(collection_name)

    def delete_collection(self,collection_name):
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"删除失败-{e}")
            return False

    def upsert_embedding(self,collection_name:str,embeddings:np.ndarray,metadata:list[dict]|dict):
        """
        插入更新向量
        :param collection_name:
        :param embeddings:
        :param metadata:
        :return:
        """
        points = []
        points_append = points.append
        if embeddings.ndim == 2:  # 多个向量
            for i, payload in enumerate(metadata):
                points_append(
                    models.PointStruct(
                        id=uuid.uuid4(),
                        vector=embeddings[i],
                        payload=payload
                    )
                )

        elif embeddings.ndim == 1:
            points_append(
                models.PointStruct(
                    id=uuid.uuid4(),
                    vector=embeddings,
                    payload=metadata
                )
            )
        else:
            raise ValueError("嵌入应该为二维或一维！！！")
        self.client.upload_points(collection_name, points=points)

    def search_embedding(self,collection_name:str,embeddings:np.ndarray,limit:int=10,filter_condition: Optional[models.Filter] = None):

        if embeddings.ndim == 2:
            results = []
            for i in range(embeddings.shape[0]):
                points=self.client.query_points(
                    collection_name=collection_name,
                    query=embeddings[i].tolist(),
                    limit=limit,
                    with_payload=True,
                    query_filter=filter_condition
                ).points
                results.append([{"相似度得分":point.score,"元数据":point.payload} for point in points])
            return results
        elif embeddings.ndim == 1:
            points=self.client.query_points(
                collection_name=collection_name,
                query=embeddings.tolist(),
                limit=limit,
                with_payload=True,
                query_filter=filter_condition
            ).points
            return [{"相似度得分":point.score,"元数据":point.payload} for point in points]
        else:
            raise ValueError(f"{embeddings}应为一维或二维！！！")






