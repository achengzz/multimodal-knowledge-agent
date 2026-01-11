#构建知识库，传入文件夹路径，
from PIL import Image
from src.storage.manage_vector_storage import VectorStoreManager
from src.encoders.multimodal_encoder import AltCLIPEncoder
from src.encoders.text_encoder import BGEEncoder
from src.utils.chunk import chunk_by_zh_chars,chunk_by_paragraph
from src.utils.parse import read_pdf, read_docx

from src.utils.read_config import config


class KnowledgeBase:
    def __init__(self, location=config.get("vector_database").get('url'), multimodal_encoder=AltCLIPEncoder, chunker=chunk_by_zh_chars, text_encoder=BGEEncoder):
        self.storage = VectorStoreManager(location)
        self.multimodal_encoder=multimodal_encoder  #多模态编码模型
        self.chunker=chunker                #文本分块器
        self.text_encoder=text_encoder      #文本编码器

    def build(self,payloads:list[dict]):
        files_payloads={"image":[],"document":[],"text":[]}
        for payload in payloads:
            files_payloads[payload['category']].append(payload)

        self.images_embedded(files_payloads["image"])
        self.documents_embedded(files_payloads["document"])
        self.texts_embedded(files_payloads["text"])
        print("构建成功！！！")

    def images_embedded(self,payloads:list[dict]):
        paths = [payload['path'] for payload in payloads]
        images = [Image.open(image) for image in paths]
        images_embeddings = self.multimodal_encoder().encode_image(images=images)
        if self.storage.collection_exist("图片库"):
            self.storage.delete_collection("图片库")
        self.storage.create_collection(collection_name="图片库",embeddings_size=768,metadata={"description":"用于存储图片的向量集，每个点就是一张图片"})
        self.storage.upsert_embedding(collection_name="图片库",embeddings=images_embeddings,metadata=payloads)
        self.storage.get_collection_info("图片库")

    def texts_embedded(self,payloads:list[dict]):
        for payload in payloads:
            name = payload['name']
            chunks = self.chunker(payload['path'])
            metadata=[{"content": chunk} for chunk in chunks]
            texts_embeddings=self.text_encoder().encode(chunks)
            if self.storage.collection_exist(name):
                self.storage.delete_collection(name)
            self.storage.create_collection(name,1024,metadata=payload)
            self.storage.upsert_embedding(name,texts_embeddings,metadata)
            self.storage.get_collection_info(name)

    def documents_embedded(self,payloads:list[dict]):
        for payload in payloads:
            name = payload['name']
            if payload['extension']=="pdf":
                extracted_text = read_pdf(payload['path'])
            elif payload['extension']=="docx":
                extracted_text = read_docx(payload['path'])
            chunks = self.chunker(extracted_text)
            metadata=[{"content":chunk} for chunk in chunks]
            documents_embeddings=self.text_encoder().encode(chunks)
            if self.storage.collection_exist(name):
                self.storage.delete_collection(name)
            self.storage.create_collection(name,1024,metadata=payload)
            self.storage.upsert_embedding(name,documents_embeddings,metadata)
            self.storage.get_collection_info(name)


    def retrieve_text(self,collection_name,query_text):

        query_embedding = self.text_encoder().encode(query_text)
        results = self.storage.search_embedding(collection_name, query_embedding[0])

        return [(result['相似度得分'], result['元数据']['content']) for result in results]

    def retrieve_image_path(self,query_text)->list[tuple]:

        query_embedding = self.multimodal_encoder().encode_text(query_text)
        results = self.storage.search_embedding("图片库", query_embedding)

        return [(result['相似度得分'], result['元数据']['path']) for result in results]


    def browse_all_knowledges(self)->list:
        return self.storage.get_all_collections()

    def get_information(self,collection_name:str)->dict:
        info=self.storage.get_collection_info(collection_name)
        return {
            "向量集名":collection_name,
            "元数据": info.config.metadata,
            "向量数量":info.points_count,
            "向量维度":info.config.params.vectors.size
        }

    def add_knowledge(self,payload):
        if payload['category']=="image":
            path=payload['path']
            images = [Image.open(path)]
            images_embeddings = self.multimodal_encoder().encode_image(images=images)
            self.storage.upsert_embedding(collection_name="图片库", embeddings=images_embeddings, metadata=payload)
            self.storage.get_collection_info("图片库")

        elif payload['category']=="document":
            name = payload['name']
            if payload['extension']=="pdf":
                extracted_text = read_pdf(payload['path'])
            elif payload['extension']=="docx":
                extracted_text = read_docx(payload['path'])
            chunks = self.chunker(extracted_text)
            metadata = [{"content": chunk} for chunk in chunks]
            documents_embeddings = self.text_encoder().encode(chunks)
            if self.storage.collection_exist(name):
                self.storage.delete_collection(name)
            self.storage.create_collection(name, 1024, metadata=payload)
            self.storage.upsert_embedding(name, documents_embeddings, metadata)
            self.storage.get_collection_info(name)

        elif payload['category']=="text":
            name = payload['name']
            chunks = self.chunker(payload['path'])
            metadata = [{"content": chunk} for chunk in chunks]
            texts_embeddings = self.text_encoder().encode(chunks)
            if self.storage.collection_exist(name):
                self.storage.delete_collection(name)
            self.storage.create_collection(name, 1024, metadata=payload)
            self.storage.upsert_embedding(name, texts_embeddings, metadata)
            self.storage.get_collection_info(name)

        else:
            raise ValueError("添加失败")



