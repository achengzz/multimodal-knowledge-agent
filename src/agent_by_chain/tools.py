
from langchain.tools import tool

from src.retrieval.query import get_all_filenames, query_image_path, query_relevant_texts


@tool
def get_embedded_filenames()->list[str]:
    """
    获取所有已建立向量索引的文件名称列表。每个文本文件对应一个独立的向量集（以其文件名命名），而所有图片则统一存储在名为“图片库”的向量集中。
    :return: 向量集名称列表，每个名称代表一个已嵌入的文件或图片库
    """
    return get_all_filenames()

@tool
def get_image_path(description:str)->list[str]:
    """
    根据自然语言描述检索相关图片的本地路径。函数将返回与描述语义最相关的3张图片的路径。
    :param description:图片内容描述，例如“一只小猫”、“价格走势折线图”、“京剧表演剧照”
    :return:图片文件路径列表，按相关度排序，最多返回3条结果
    """
    return query_image_path(description)

@tool
def get_relevant_texts(file_name:str,query_text:str)->list[str]:
    """
    在指定文件的向量索引中检索与查询文本语义相关的内容片段。
    :param file_name: 目标文件名（须为已嵌入的文本文件）
    :param query_text: 查询语句或关键词
    :return:相关文本片段列表，按相关度排序
    """
    return query_relevant_texts(file_name,query_text)


tools=[get_image_path,get_relevant_texts,get_embedded_filenames]