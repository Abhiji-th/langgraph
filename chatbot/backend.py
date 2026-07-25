from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatbotState):
    messages = state['messages']

    response = llm.invoke(messages)

    return {'messages': [response]}


checkpointer = MemorySaver()


graph = StateGraph(ChatbotState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# CONFIG = {'configurable': {'thread_id': 'thread-1'}}

# chatbot.invoke(
#                 {'messages': [HumanMessage(content='Hi')]},
#                 config = CONFIG,
# )

# response = chatbot.stream(
#                 {'messages': [HumanMessage(content='Hi')]},
#                 config = CONFIG,
#                 stream_mode='messages'
#             )

# for chunk, metadata in response:
#     if chunk.content:
#         print(chunk.content[0]['text'])


