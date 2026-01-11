

from src.agent_by_chain.agent import agent
import gradio as gr




async def interact_with_langchain_agent(prompt, messages):
    """
    异步流式交互函数，用于与智能体进行实时对话
    :param prompt:用户的输入文本:问题或指令
    :param messages:历史对话消息列表，包含所有先前的用户和ai交互记录
    :return:
    """
    messages.append({"role":"user", "content":prompt})
    yield messages,""
    async for chunk in agent.astream({"messages":messages},stream_mode="updates"):
        for role,data in chunk.items():
            new_message = data["messages"][-1]
            if role == "tools":
                # messages.append({"role":"assistant", "content":data["messages"][-1].content,"metadata":{"title": f"🛠️ 使用工具 {data['messages'][-1].name}"}})

                messages.append({
                    "role":"assistant",
                    "content":new_message.content,
                    "metadata":{"title": f"🛠️ 使用工具 {new_message.name}"},
                    # "options":{"type":"tools","name":new_message.name,"tool_call_id":new_message.tool_call_id}
                })
                yield messages,""
            if role == "model":
                messages.append({
                    "role":"assistant",
                    "content": new_message.content,
                    # "options":{"type":"model","tool_calls":new_message.tool_calls or []}
                })
                yield messages,""



with gr.Blocks(title="智能本地知识库助手",) as chat_app:
    # 1. 标题区域
    gr.Markdown(
        """
        # 🧠 具备本地知识库的智能体
        *基于LangChain构建的智能问答助手*
        """,
        elem_id="header-section"
    )

    # 2. 聊天区域
    knowledge_agent_chatbot = gr.Chatbot(
        label="智能助手对话历史",
        avatar_images=(
            r"C:\Users\Cheng\Desktop\智能体\src\ui\user.png",
            r"C:\Users\Cheng\Desktop\智能体\src\ui\ai.png",
        ),
        max_height=400,

        elem_id="chatbot-container"
    )

    # 3. 输入区域
    with gr.Row(elem_id="input-section",equal_height=True):
        user_input = gr.Textbox(
            lines=2,  # 增加行数，方便输入长文本
            label="请输入您的问题",
            placeholder="请在这里输入您的问题...",
            show_label=False,  # 隐藏标签
            elem_id="message-input",
            scale = 5,  # 文本框占用宽度

        )

        submit_btn = gr.Button(
            "发送",
            variant="primary",
            size="lg",
            elem_id="send-button",
            scale=1,

        )

    # 4. 控制区域
    with gr.Row(elem_id="control-section"):
        clear_btn = gr.Button("清空对话", variant="secondary")


    # 5. 事件绑定
    # 回车发送
    # user_input.submit(
    #     fn=interact_with_langchain_agent,
    #     inputs=[user_input, knowledge_agent_chatbot],
    #     outputs=[knowledge_agent_chatbot],
    # )


    # 按钮发送
    submit_btn.click(
        fn=interact_with_langchain_agent,
        inputs=[user_input, knowledge_agent_chatbot],
        outputs=[knowledge_agent_chatbot,user_input],
    )


    # 清空按钮事件
    def clear_chat():
        return [], None


    clear_btn.click(
        fn=clear_chat,
        outputs=[knowledge_agent_chatbot, user_input]
    )

    # 6. 额外的说明区域
    gr.Markdown(
        """
        ---
        **使用说明：**
        1. 在下方输入框输入问题
        2. 按回车或点击"发送"按钮
        3. 点击"清空对话"重置对话

        **注意事项：**
        - 回答基于本地知识库
        - 可能需要处理时间，请耐心等待
        - 对话内容仅用于当前会话
        """,
        elem_id="instructions-section"
    )



def start_chatapp():
    chat_app.launch(
        debug=False,
        show_error=True,  # 显示错误信息
        theme=gr.themes.Soft()
    )



