import time
import streamlit as st
import requests
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from qdrant_client import QdrantClient
from mistralai import Mistral

import memory
import api_key
import retriever
import database

# chat with Mistral
def chat(client, query):
    response = client.chat.stream(
        model="mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": query,
            },
        ]
    )
    for chunk in response:
        yield chunk.data.choices[0].delta.content or ""

# merge query, conversation history and Qdrant response
def merge(query, history, qdrant_response, language) :
    history = "\n".join([str(el) for el in history])
    return f"""
Tu es un assistant juridique expert en droit des brevets. Tu aides l'utilisateur à comprendre et à naviguer dans les aspects juridiques liés aux brevets (ex : dépôt, validité, contrefaçon, brevets européens, procédure à l'INPI ou à l'OEB, etc.), notamment dans le cadre de la préparation de concours. 

Tu dois conserver en mémoire tous les éléments. Pour cela, voici l'historique de la conversation ou un résumé détaillé : 
```
{history}
```

L'utilisateur a envoyé le message suivant. C'est à celui-ci, et celui-ci uniquement, que tu vas répondre dans ta prochaine réponse. Le voici :
```
{query}
```

Pour y répondre, tu pourras te baser dans un premier temps sur le texte suivant, issu d'une base de connaissances spécialisée dans le domaine :
```
{qdrant_response}
```

Quelques autres remarques très importantes :
- Tu dois IMPÉRATIVEMENT répondre dans la langue suivante, peu importe la langue dans laquelle l'utilisateur écrit ses messages : {language}. 
- Ta réponse devra être claire, précise, pédagogique, et uniquement en lien avec le droit des brevets. 
- Si le texte fourni (issue de la base de données) n'a rien à voir avec la demande de l'utilisateur, tu pourras dans ce cas y répondre par tes connaissances personnelles, mais tu devras impérativement le préciser de manière explicite à l'utilisateur. 
- Si tu n'as pas la réponse à la question de l'utilisateur, tu lui indiqueras clairement et honnêtement plutôt que de l'inventer. 
- Termine impérativement ton prompt par "**Sources** :", je mettrais les sources en-dessous (tu peux évidemment changer la langue si besoin)
"""

# initialisation
st.title("CYentists' chatbot")
if "session_started" not in st.session_state :
    Settings.embed_model = HuggingFaceEmbedding(model_name="baai/bge-small-en-v1.5")
    st.session_state.mistral_client = Mistral(api_key=api_key.MISTRAL_API_KEY)
    st.session_state.qdrant_client = QdrantClient(
        "https://3cf7f5eb-3647-41f9-94aa-6dbc76a973d4.us-west-1-0.aws.cloud.qdrant.io:6333",
        port = 6333,
        api_key = api_key.QDRANT_API_KEY
    )
    st.session_state.qdrant_db = database.Qdrant_DB(st.session_state.qdrant_client)
    st.session_state.retriever = retriever.create_full_retriever(st.session_state.qdrant_db)
    st.session_state.session_started = True
    st.session_state.bot_memory = memory.memory
    st.session_state.summarized_history = memory.history
    st.session_state.complete_history = []
    st.session_state.langue = "Français"
    st.session_state.previous_retriever_name = ""
    
# display messages
for msg in st.session_state.complete_history :
    with st.chat_message(msg["role"]) :
        st.markdown(msg["content"])

# option buttons (select languages, select retriever and clear the conversation)
st.sidebar.selectbox("Choix de la langue", ["Français", "English", "Deutsch"], key = "language")
st.sidebar.selectbox("Choix du retriever", ["Embedding/BM25 retrievers + rerank", "Embedding retriever", "BM25 retriever"], key = "retriever_name")
st.sidebar.slider("Top-K pour l'Embedding Retriever", min_value=2, max_value=150, value=50, key="top_k_embedding")
st.sidebar.slider("Top-K pour le BM25 Retriever", min_value=2, max_value=30, value=15, key="top_k_bm25")
st.sidebar.slider("Top-N pour le Reranker", min_value=2, max_value=10, value=6, key="top_n_rerank")
show_scores = st.sidebar.checkbox("Afficher les scores de pertinence des sources", value=False)

if st.sidebar.button("Effacer la conversation") :
    st.session_state.bot_memory = memory.memory
    st.session_state.summarized_history = memory.history
    st.session_state.complete_history = []
    st.rerun()

# chat with the LLM
query = st.chat_input("Say something...")
if query is not None and query.strip() != "" :

    with st.chat_message("user") :
        st.markdown(query)

    if (
        st.session_state.previous_retriever_name != st.session_state.retriever_name or
        st.session_state.previous_top_k_embedding != st.session_state.top_k_embedding or
        st.session_state.previous_top_k_bm25 != st.session_state.top_k_bm25 or
        st.session_state.previous_top_n_rerank != st.session_state.top_n_rerank
    ):
        st.session_state.retriever = retriever.create_chosen_retriever(
            st.session_state.qdrant_db, 
            retriever_name=st.session_state.retriever_name, 
            emb_k=st.session_state.top_k_embedding, 
            bm25_k=st.session_state.top_k_bm25, 
            reranker_n=st.session_state.top_n_rerank
        ) 
        
        st.session_state.previous_retriever_name = st.session_state.retriever_name
        st.session_state.previous_top_k_embedding = st.session_state.top_k_embedding
        st.session_state.previous_top_k_bm25 = st.session_state.top_k_bm25
        st.session_state.previous_top_n_rerank = st.session_state.top_n_rerank
    
        st.write(f"Retriever mis à jour : {st.session_state.retriever_name} (Embedding k={st.session_state.top_k_embedding}, BM25 k={st.session_state.top_k_bm25}, Reranker n={st.session_state.top_n_rerank})")

    
    context, sources = retriever.retrieve_context_and_sources(st.session_state.retriever, query, show_scores)
    str_sources = "\n" + "\n".join(f"- {src}" for src in sources)
    prompt = merge(query, st.session_state.summarized_history, context, st.session_state.language)
    time.sleep(1.1)
    
    with st.chat_message("assistant"):
        full_response = ""
        resp_area = st.empty()
        for token in chat(st.session_state.mistral_client, prompt):
            full_response += token
            resp_area.markdown(full_response)
        for token in str_sources :
            full_response += token
            resp_area.markdown(full_response)
            time.sleep(0.02)
        mistral_response = full_response
    
    st.session_state.bot_memory, st.session_state.summarized_history = memory.update_memory(
        st.session_state.bot_memory, 
        ChatMessage(role = "user", content = query), 
        ChatMessage(role = "assistant", content = str(mistral_response))
    )
    
    st.session_state.complete_history.append({"role" : "user", "content" : query})
    st.session_state.complete_history.append({"role" : "assistant", "content" : mistral_response})