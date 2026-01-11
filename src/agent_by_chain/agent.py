from pprint import pprint

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent_by_chain.tools import tools


load_dotenv(override=True)
llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")


from langchain.agents import create_agent

agent=create_agent(
    model=llm,
    tools=tools
)
if __name__ == '__main__':

    messages=[SystemMessage(content="你是一个有用的助手"),HumanMessage(content=input("请输入提示词:\n"))]
    while True:
        for chunk in agent.stream({"messages":messages},stream_mode="updates"):
            print(chunk)
            for step, data in chunk.items():
                print(step)
                print(data)
                print("*"*60)

                # print(f"step: {step}")
                # print(f"content: {data['messages'][-1].content_blocks}")
                # messages.append(data["messages"][-1])

        user_inputs=input("请输入提示词:\n")
        if user_inputs=="quit" or user_inputs=="exit":
            break
        messages.append(HumanMessage(content=user_inputs))


