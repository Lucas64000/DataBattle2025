from llama_index.core import Settings
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.mistralai import MistralAI

import api_key

# initialisation
from llama_index.llms.mistralai import MistralAI
Settings.llm = MistralAI(api_key=api_key.MISTRAL_API_KEY)

# create memory
memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_history = [],
    llm = Settings.llm,
    token_limit = 300
)
history = memory.get()

# update memory
def update_memory(memory, *args) :
    for new_chat in args :
        memory.put(new_chat)
    return memory, memory.get()