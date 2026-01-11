



from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.state import AgentState

from src.agent.tools import tools
from src.agent.schemas import QueryAnalysisResult
from src.agent.prompts import ANALYSIS_SYSTEM_PROMPT


# load_dotenv(override=True)
# llm = init_chat_model(model="deepseek-chat", model_provider="deepseek")

def create_initial_state(user_input: str) -> AgentState:
    """创建代理初始状态"""
    return {
        "messages": [HumanMessage(content=user_input)],
        "original_input": user_input,
        "optimized_request": None,      # 分析前为空

    }



# def analyze_query(state:AgentState,llm:BaseChatModel)->AgentState:
#     """分析节点，使用llm分析用户输入，将用户的原始请求优化为优化后的清晰、可执行的请求"""
#
#     if state.get("optimized_request") is None:
#         prompt=f"用户的原始输入：{state.get('original_input')}"
#         llm_with_structured=llm.with_structured_output(schema=QueryAnalysisResult)
#         schema_response=llm_with_structured.invoke([SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),HumanMessage(content=prompt)])
#         optimized_request=schema_response.optimized_prompt
#     else:
#         optimized_request = state.get("optimized_request")
#
#     llm_with_tools=llm.bind_tools(tools=tools)
#     response=llm_with_tools.invoke([HumanMessage(content=optimized_request)])
#     return {
#         "messages":[response],
#         "original_input":state.get("original_input"),
#         "optimized_request":optimized_request,
#     }


def optimize_request(state:AgentState,llm:BaseChatModel)->AgentState:
    print("正在执行优化节点...")
    """将用户的原始请求优化成清晰，无歧义的请求"""
    prompt = f"用户的原始输入：{state.get('original_input')}"
    llm_with_structured = llm.with_structured_output(schema=QueryAnalysisResult)
    schema_response = llm_with_structured.invoke([SystemMessage(content=ANALYSIS_SYSTEM_PROMPT), HumanMessage(content=prompt)])
    optimized_request = schema_response.optimized_prompt

    print(f"优化结果：{optimized_request}")

    return {
        "messages":[HumanMessage(content=optimized_request)],
        "original_input":state.get("original_input"),
        "optimized_request":optimized_request,
    }



# def rewrite_query(state:AgentState,llm:BaseChatModel)->AgentState:
#     """检索工具可能会返回潜在的不相关文档，这表明需要改进原始用户问题"""
#     user_query=state.get("original_input")
#     prompt=f"""请深入分析用户的原始查询，理解其**底层语义意图和真实需求**，然后重新构建一个**更清晰、更完整、更易执行**的问题。\n
#                 用户的原始查询：{user_query}\n
#                 你应该返回的是优化后的用户请求，而不是直接回复用户"""
#     llm_with_structured = llm.with_structured_output(schema=QueryAnalysisResult)
#     response = llm_with_structured.invoke([HumanMessage(content=prompt)])
#     hm_msg=HumanMessage(content=f"用户的请求已优化：{response.optimized_prompt}")
#     return {
#         "messages":[hm_msg],
#         "original_input":state.get("original_input"),
#         "optimized_request":response.optimized_prompt,
#     }




def summarize_answers(state:AgentState,llm:BaseChatModel)->AgentState:
    print("正在归纳总结...")
    user_query = state.get("original_input")
    context = state.get("messages")[-1].content
    prompt = f"""你是一个专业的问答助手。请基于以下检索到的上下文信息来回答问题。

                回答要求：
                1. **准确可靠**：严格基于提供的上下文信息
                2. **简洁明了**：用3-5句话清晰表达
                3. **重点突出**：直接回答核心问题
                4. **诚实可信**：如果信息不足，如实告知

                用户问题：{user_query}

                参考上下文：{context}
                现在请基于上下文生成回答："""
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "messages":[response],
        "original_input":state.get("original_input"),
        "optimized_request":None
    }

def reasoning(state:AgentState,llm:BaseChatModel)->AgentState:
    """思考"""
    print("正在思考...")
    llm_with_tools = llm.bind_tools(tools)
    response=llm_with_tools.invoke(state["messages"] or state["optimized_request"])
    print(f"思考和行动：{response.content},准备调用工具：{response.tool_calls}")
    return {
        "messages":[response],
        "original_input":state.get("original_input"),
        "optimized_request":state.get("optimized_request"),
    }

