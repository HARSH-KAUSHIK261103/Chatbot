import streamlit as st
from llama_index.core import Document, VectorStoreIndex
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import PGRetriever
from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def init_page() -> None:
    st.set_page_config(page_title="Personal Chatbot")
    st.header("Personal Chatbot")

def select_llm() -> LlamaCPP:
    return LlamaCPP(
        model_path="C:/Users/HarshKaushik/chat/llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=500,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

def init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply to your answer in markdown format."
            )
        ]

def create_index() -> VectorStoreIndex:
    documents = [
        Document(text="Cardiology is a branch of medicine that deals with the disorders of the heart and the blood vessels. It includes diagnosis and treatment of congenital heart defects, coronary artery disease, heart failure, and valvular heart disease."),
        Document(text="Diabetes mellitus is a group of metabolic disorders characterized by a high blood sugar level over a prolonged period. Symptoms often include frequent urination, increased thirst, and increased appetite."),
        Document(text="Hypertension, also known as high blood pressure, is a condition in which the force of the blood against the artery walls is too high. It can lead to severe health complications and increase the risk of heart disease, stroke, and sometimes death."),
        Document(text="Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body. Treatments include surgery, chemotherapy, radiation therapy, and immunotherapy."),
        Document(text="Vaccination is the administration of a vaccine to help the immune system develop protection from a disease. Vaccines have played a critical role in eradicating diseases like smallpox and controlling outbreaks of measles, polio, and influenza"),
        Document(text="Medical imaging is a technique used to create visual representations of the interior of a body for clinical analysis and medical intervention. Techniques include X-ray, MRI, CT scans, and ultrasound."),
        Document(text="Mental health disorders encompass a wide range of conditions that affect mood, thinking, and behavior. Common disorders include depression, anxiety disorders, schizophrenia, eating disorders, and addictive behaviors"),
        Document(text="Emergency medicine is the medical specialty concerned with the care of illnesses or injuries requiring immediate medical attention. It involves the diagnosis and treatment of acute conditions in a hospital's emergency department."),
        Document(text="Neurology is the branch of medicine dealing with disorders of the nervous system. Neurologists diagnose and treat diseases and conditions such as epilepsy, Parkinson's disease, multiple sclerosis, and migraines."),
        Document(text="Nutrition is the science that interprets the nutrients and other substances in food in relation to maintenance, growth, reproduction, health, and disease of an organism. Proper nutrition is crucial for preventing chronic diseases and promoting overall health.")
    ]
    embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    return index

def create_retriever(index: VectorStoreIndex) -> PGRetriever:
    retriever = PGRetriever(index=index, sub_retrievers=[] ,verbose=True)
    return retriever

def retrieve_documents(retriever, query: str, top_k: int = 3):
    results = retriever.retrieve(query)
    return results

def generate_response_with_context(llm, query: str, documents):
    context = " ".join([doc.text for doc in documents])
    input_text = f"Context: {context}\nQuestion: {query}"
    response = llm.complete(input_text)
    
    # Handle different response formats
    if hasattr(response, 'text'):
        return response.text.strip()
    elif hasattr(response, 'get_text'):
        return response.get_text().strip()
    else:
        raise ValueError("Unknown response format.")

def main() -> None:
    init_page()
    llm = select_llm()
    init_messages()
    index = create_index()
    retriever = create_retriever(index)

    if user_input := st.text_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):
            retrieved_docs = retrieve_documents(retriever, user_input)
            answer = generate_response_with_context(llm, user_input, retrieved_docs)
            st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            st.markdown(f"**Assistant**: {message.content}")
        elif isinstance(message, HumanMessage):
            st.markdown(f"**User**: {message.content}")

if __name__ == "__main__":
    main()
