from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.constants import END

from src.agent.schemas import CorrelationAnalysisResult
from src.agent.state import AgentState





def manual_review_condition(state:AgentState,llm:BaseChatModel):
    """人工审核，机器辅助"""
    user_query = state.get("original_input")
    context = state.get("messages")[-1].content
    prompt = f"""你是一个文档相关度评估专家，你的任务是评估检索到的文档与用户问题的相关程度，并提供详细的评估结果。结果是一个0-10之间的整数，值越大，表示越相关\n
                用户问题：{user_query}\n
                检索到的内容：{context}"""

    llm_with_structured = llm.with_structured_output(schema=CorrelationAnalysisResult)
    response = llm_with_structured.invoke([HumanMessage(prompt)])
    result=input(f"智能体根据你的问题或请求给出了回答和置信分（满分10分）：\n    {context}\n置信分{response.rate}\n你是否同意该结果？（同意：y  拒绝：n）\n")
    if result=="y":
        return END
    else:
        return "optimize_request"





