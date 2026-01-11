from langchain.agents.middleware import AgentMiddleware, ModelResponse, ModelRequest
from langchain.agents.middleware.types import ModelCallResult, before_model, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any, Callable

from langgraph.typing import StateT,ContextT


class MyMiddleware(AgentMiddleware):
    """自定义中间件"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
            """agent开始前执行"""
            print("智能体  开始前执行")
            return None

        def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
            """agent结束后执行"""
            print("智能体  结束后执行")
            return None

        def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
            """model开始前执行"""
            print("模型   开始前执行")
            return None

        def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
            """model结束后执行"""
            print("模型  结束后执行")
            return None

        def wrap_model_call(
                self,
                request: ModelRequest,
                handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            """包裹模型调用"""
            return handler(request)


class TokenCounter:
    """Token计数器"""

    def __init__(self):
        self.total = 0

    def __call__(self, state: AgentState, runtime: Runtime):
        """直接作为可调用对象使用"""
        last_message = state["messages"][-1]

        if hasattr(last_message, "response_metadata"):
            tokens = last_message.response_metadata.get('token_usage', {}).get('total_tokens', 0)
            self.total += tokens
            print(f"本次: {tokens} tokens | 总计: {self.total} tokens")

        return state


@before_model(can_jump_to=["end"])
def check_message_limit(state:AgentState,runtime:Runtime):
    if len(state["messages"])>=50:
        return {"messages":[AIMessage("对话已达到上限,请开启新对话")],
                "jump_to":"end"}