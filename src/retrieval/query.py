from src.retrieval.rerank import Reranker
from src.storage.build_knowledge_base import KnowledgeBase


def query_image_path(query_text:str,limit:int=3)->list[str]:
    """
    输入图片描述，返回该图片的位置
    :param query_text: 描述文字
    :param limit: 结果限制
    :return:
    """
    knb=KnowledgeBase(location="http://192.168.10.10:6333")
    results=knb.retrieve_image_path(query_text)[:limit]
    return [result[1] for result in results]



def query_relevant_texts(file_name:str,query_text:str,limit=5)->list[str]:
    """
    获取与查询语句最相关的limit个文本，用于rag
    :param file_name: 文件名
    :param query_text: 查询文字
    :param limit: 结果限制
    :return:
    """
    reranker = Reranker()
    knb = KnowledgeBase(location="http://192.168.10.10:6333")
    results=knb.retrieve_text(file_name, query_text)
    texts=[result[1] for result in results]
    results,scores=reranker.rerank(query_text,texts)
    return results[:limit]

def get_all_filenames():
    """
    获取知识库中的所有文件名
    :return:
    """
    knb=KnowledgeBase(location="http://192.168.10.10:6333")
    results= knb.browse_all_knowledges()
    return results


