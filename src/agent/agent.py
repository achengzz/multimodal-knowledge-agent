from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from src.agent.graphs import main_graph
from src.agent.nodes import create_initial_state

load_dotenv(override=True)



def chat_with_agent():
    """函数式调用"""
    llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")
    user_input=input("请输入你的问题：\n")
    state=create_initial_state(user_input)

    agent=main_graph(llm=llm)
    response=agent.invoke(state)
    print(f"智能体执行结果：\n{response.get('messages')[-1].content}")


class AgentWithKnowledgeBase:
    """类封装"""
    def __init__(self,llm,tools=None):
        self.llm=llm
        self.tools=tools            #这个可以自己传入工具列表，稍微修改逻辑让智能体可调用，由于我在构建图的时候就传入了，这里就直接为空
        self.graph=main_graph(llm=llm)

    def query(self,request):
        """请输入你的请求或问题"""
        state=create_initial_state(request)
        response=self.graph.invoke(state)
        print(f"智能体执行结果：\n{response.get('messages')[-1].content}")
        return {
            "messages":response.get('messages')
        }



if __name__ == '__main__':
    # chat_with_agent()

    agent=AgentWithKnowledgeBase(llm=init_chat_model(model="deepseek-chat", model_provider="deepseek"))
    agent.query(input("请输入你的请求或问题：\n"))