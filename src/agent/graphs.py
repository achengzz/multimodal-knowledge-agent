from langchain_core.language_models import BaseChatModel

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from src.agent.edges import  manual_review_condition
from src.agent.state import AgentState
from src.agent.nodes import optimize_request,summarize_answers, reasoning
from src.agent.tools import tools






def react_subgraph(llm:BaseChatModel):

    workflow=StateGraph(AgentState)
    workflow.add_node("reasoning", lambda state:reasoning(state,llm))
    workflow.add_node("acting",ToolNode(tools=tools,name="acting"))

    workflow.add_edge(START,"reasoning")
    workflow.add_conditional_edges(
        "reasoning",
        tools_condition,
        {
            "tools":"acting",
            END:END
        }
    )
    workflow.add_edge("acting","reasoning")
    graph=workflow.compile()
    return graph



def main_graph(llm:BaseChatModel):
    workflow=StateGraph(AgentState)

    workflow.add_node("optimize_request",lambda state:optimize_request(state,llm))
    workflow.add_node("react",react_subgraph(llm))
    workflow.add_node("summarize_answers",lambda state:summarize_answers(state,llm))

    workflow.add_edge(START,"optimize_request")
    workflow.add_edge("optimize_request","react")
    workflow.add_edge("react","summarize_answers")

    workflow.add_conditional_edges(
        "summarize_answers",
        lambda state:manual_review_condition(state,llm),
        {
            END:END,
            "optimize_request":"optimize_request"
        }

    )
    graph=workflow.compile()
    return graph
