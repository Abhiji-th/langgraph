import streamlit as st
from backend import chatbot
from langchain.messages import HumanMessage

thread_id = 1 
CONFIG = {'configurable': {'thread_id': thread_id}}


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_message = st.chat_input('Type here')

if user_message:

    st.session_state['message_history'].append({'role': 'user', 'content': user_message})
    with st.chat_message('user'):
        st.text(user_message)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_message)]},
                config = CONFIG,
                stream_mode='messages'
            )
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})                                           