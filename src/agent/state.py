
from typing import TypedDict,Annotated
from langgraph.graph import add_messages





class AgentState(TypedDict):
    """统一代理状态（包含所有阶段需要的字段）"""
    messages: Annotated[list, add_messages]
    original_input: str
    optimized_request: str|None
