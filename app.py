import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# -----------------------------
# 1. 임베딩 모델 (CPU로 충분히 빠름)
# -----------------------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 2. 간단 문서 (테스트용)
# -----------------------------
docs = [
    Document(page_content="AWS EC2는 아마존 클라우드의 가상 서버입니다."),
    Document(page_content="Docker는 컨테이너 기반 가상화 플랫폼입니다."),
    Document(page_content="쿠버네티스는 컨테이너 오케스트레이션 도구입니다."),
]

db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

# -----------------------------
# 3. GPT-OSS-20B LLM 설정
# -----------------------------
model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,  # Mac MPS GPU 사용 가능
    device_map="auto"
)

# pipeline을 이용해 Harmony response format 적용
hf_pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipe)

# -----------------------------
# 4. PromptTemplate 생성
# -----------------------------
prompt_template = """다음 문서를 참고하여 질문에 대한 답을 한국어로 간결하게 작성하세요.
문서를 그대로 반복하지 말고, 질문에 대해 핵심만 답하세요.

Context:
{context}

Question: {question}
Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -----------------------------
# 5. RetrievalQA 체인
# -----------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}  # 문자열 대신 PromptTemplate 사용
)

# -----------------------------
# 6. Streamlit UI
# -----------------------------
st.title("🔎 Hugging Face RAG 챗봇 (GPT-OSS-20B)")
user_q = st.text_input("질문을 입력하세요:")

if user_q:
    answer = qa.run(user_q)
    st.write("🤖 답변:", answer)
