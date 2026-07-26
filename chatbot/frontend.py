import streamlit as st
from backend import chatbot, get_threads
from langchain.messages import HumanMessage
import uuid

############################### Util Functions #############################
def stream_response(user_message, config):
    for chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_message)]},
                config = config,
                stream_mode='messages'
    ):
        if not chunk.content:
            continue

        text = chunk.content[0].get("text", "")

        if text:
            yield text

def generateThreadID():
    return str(uuid.uuid4())

def resetChat():
    st.session_state['message_history'] = []
    thread_id = generateThreadID()
    st.session_state['thread_id'] = thread_id
    addThread(thread_id)

def addThread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads']. append(thread_id)

def loadChat(thread_id):
    st.session_state['thread_id'] = thread_id

    config = {'configurable': {'thread_id': thread_id}}
    state = chatbot.get_state(config=config)

    messages = []
    if state.values:
        messages = state.values['messages']

    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
            content = msg.content
        else:
            role = 'assistant'
            content = msg.content[0]['text']

        temp_messages.append({'role': role, 'content': content})

    st.session_state['message_history'] = temp_messages
 

############################### Session state ################################
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_threads()

if 'thread_id' not in st.session_state:
    thread_id = generateThreadID()

    st.session_state['thread_id'] = thread_id
    addThread(thread_id)

############################### Chat Window #####################################
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

############################### Chat Input ##################################
user_message = st.chat_input('Type here')


############################### Side bar ####################################
st.sidebar.title('My chatbot')

if st.sidebar.button('New chat'):
    resetChat()
    st.rerun()

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id):
        loadChat(thread_id)
        st.rerun()

CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

if user_message:

    st.session_state['message_history'].append({'role': 'user', 'content': user_message})
    with st.chat_message('user'):
        st.text(user_message)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            stream_response(user_message, CONFIG)
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})                                           