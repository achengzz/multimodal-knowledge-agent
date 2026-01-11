from src.storage.build_knowledge_base import KnowledgeBase
from src.storage.generate_file_payload import PayloadBuilder
# from src.ui.ui import start_chatapp

from src.agent.agent import chat_with_agent
from src.utils.read_config import config


def build_knowledge_base():
    """构建知识库，执行一次即可"""
    payloads=PayloadBuilder().search_files(config.get("knowledge_base").get('path'))
    knb = KnowledgeBase(config.get("vector_database").get('url'))
    knb.build(payloads)         #如果添加不在此文件夹下的文件，可调用add_knowledge方法添加




if __name__ == '__main__':
    chat_with_agent()
    # start_chatapp()


