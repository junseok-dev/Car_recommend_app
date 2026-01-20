import streamlit as st
import os
from dotenv import load_dotenv

# [Fix] 최신 패키지 구조 반영 (2026 표준)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_core.documents import Document

# [Fix] 노란 줄 해결: RetrievalQA 대신 최신 체인 생성 도구 사용
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 0. 환경 변수 로드 ---
load_dotenv()


# .env 파일이나 Streamlit Secrets에서 API 키 로드
def get_api_key():
    """Streamlit Secrets 또는 .env 파일에서 API 키 로드"""
    # 1. Streamlit Secrets 확인 (배포 환경)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]

    # 2. 환경 변수 확인 (.env 파일)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 3. 없으면 UI에서 입력받기
    return None


api_key = get_api_key()

# --- 1. 앱 설정 ---
st.set_page_config(page_title="Global Car AI 소믈리에", layout="wide")
st.title("🏎️ 전 세계 자동차 추천 RAG 시스템")


# --- 2. RAG 시스템 초기화 (절대 경로 로직 적용) ---
@st.cache_resource
def load_car_data():
    """CSV 데이터 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "data", "Cars_Datasets_2025.csv")

    if not os.path.exists(file_path):
        st.error(f"⚠️ 파일을 찾을 수 없습니다!")
        return None

    encoding_list = ["utf-8", "euc-kr", "cp949", "latin-1", "iso-8859-1"]
    df = None

    for encoding in encoding_list:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except:
            continue

    return df


@st.cache_resource
def init_rag_system():
    """RAG 시스템 초기화"""
    # 현재 실행 중인 app.py 파일의 절대 경로를 가져옵니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # [수정] 사용자가 바꾼 파일 이름과 data 폴더 경로를 결합합니다.
    file_path = os.path.join(current_dir, "data", "Cars_Datasets_2025.csv")

    # 파일 존재 여부 확인 및 디버깅 메시지
    if not os.path.exists(file_path):
        st.error(f"⚠️ 파일을 찾을 수 없습니다!")
        st.info(f"현재 앱이 찾고 있는 경로: {file_path}")
        # 혹시 몰라 현재 폴더 구조를 출력해줍니다.
        if os.path.exists(os.path.join(current_dir, "data")):
            st.write(
                "data 폴더 내 파일들:", os.listdir(os.path.join(current_dir, "data"))
            )
        else:
            st.write("data 폴더 자체가 존재하지 않습니다.")
        return None

    try:
        # Step 1: 데이터 로드 (인코딩 자동 감지)
        # 여러 인코딩 시도
        encoding_list = ["utf-8", "euc-kr", "cp949", "latin-1", "iso-8859-1"]
        df = None

        for encoding in encoding_list:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                st.info(f"✅ 인코딩: {encoding}으로 로드됨")
                break
            except:
                continue

        if df is None:
            st.error("CSV 파일을 읽을 수 없습니다. 지원하지 않는 인코딩입니다.")
            return None

        # 문서로 변환
        documents = []
        for idx, row in df.iterrows():
            # 각 행을 텍스트로 변환
            content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            doc = Document(page_content=content, metadata={"index": idx})
            documents.append(doc)

        # Step 2: 문서 분할 및 임베딩
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=api_key
        )

        # 벡터 DB 생성 (FAISS)
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {e}")
        return None


# --- 3. 메인 로직 ---
if not api_key:
    st.warning("⚠️ API 키가 필요합니다")
    st.info(
        """
    **로컬 실행:** `.env` 파일에 `OPENAI_API_KEY=sk-...` 추가
    
    **Streamlit Cloud 배포:** 
    1. 리포지토리의 `Settings` → `Secrets` 이동
    2. 다음 내용 추가:
    ```
    OPENAI_API_KEY = "sk-..."
    ```
    """
    )

    # UI에서 직접 입력 받기 (테스트용)
    api_key_input = st.text_input(
        "🔑 또는 여기에 API 키를 입력하세요 (테스트용):", type="password"
    )
    if api_key_input:
        api_key = api_key_input
    else:
        st.stop()

# API 키가 있으면 앱 실행
if api_key:
    with st.spinner("자동차 데이터를 불러오는 중입니다..."):
        vectorstore = init_rag_system()
        df = load_car_data()

    if vectorstore and df is not None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

        system_prompt = (
            "당신은 자동차 전문 상담원입니다. 검색된 문맥(Context)을 사용하여 질문에 답하세요."
            "한국어로 답변하되, 모델명은 영어로 적어주세요."
            "\n\n"
            "{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # LCEL 방식으로 RAG 체인 구성
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retriever = vectorstore.as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # 채팅 UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # --- 사용자 입력 UI (조건 선택) ---
        st.markdown("### 🔍 차량 추천 조건 선택")

        col1, col2 = st.columns(2)

        with col1:
            # 회사 선택
            company = st.selectbox(
                "🏢 자동차 제조사 (선택사항)",
                ["전체"] + list(df["Company Names"].unique()),
                help="특정 제조사를 선택하세요",
            )

            # 연료 타입 선택
            fuel_type = st.multiselect(
                "⛽ 연료 타입 (선택사항)",
                df["Fuel Types"].unique(),
                help="복수 선택 가능합니다",
            )

            # 엔진 타입 선택
            engine = st.multiselect(
                "🔧 엔진 타입 (선택사항)",
                df["Engines"].unique(),
                help="복수 선택 가능합니다",
            )

        with col2:
            # 좌석 수 선택
            seats = st.selectbox(
                "🪑 좌석 수 (선택사항)",
                ["전체"]
                + sorted([int(s) for s in df["Seats"].unique() if str(s).isdigit()]),
                help="원하는 좌석 수를 선택하세요",
            )

            # 최대 예산 입력
            max_price = st.text_input(
                "💰 최대 예산 (선택사항)",
                placeholder="예: $50,000",
                help="예산을 입력하세요 (예: $100,000)",
            )

            # 성능 선호도
            performance_pref = st.text_input(
                "⚡ 성능 선호도 (선택사항)",
                placeholder="예: 빠른 가속, 높은 최고속도",
                help="원하는 성능 특성을 입력하세요",
            )

        # 추가 요구사항
        additional_notes = st.text_area(
            "📝 추가 요청사항 (선택사항)",
            placeholder="예: 럭셔리하고 조용한 차, 패밀리카, 스포츠카 등",
            height=80,
            help="기타 요구사항을 자유롭게 입력하세요",
        )

        # 추천 버튼
        if st.button("🚗 차량 추천받기", use_container_width=True, type="primary"):
            # 선택된 조건을 바탕으로 쿼리 생성
            query_parts = []

            if company != "전체":
                query_parts.append(f"제조사: {company}")

            if fuel_type:
                query_parts.append(f"연료: {', '.join(fuel_type)}")

            if engine:
                query_parts.append(f"엔진: {', '.join(engine)}")

            if seats != "전체":
                query_parts.append(f"좌석: {seats}명")

            if max_price:
                query_parts.append(f"최대 예산: {max_price}")

            if performance_pref:
                query_parts.append(f"성능: {performance_pref}")

            if additional_notes:
                query_parts.append(f"추가 요청: {additional_notes}")

            user_input = (
                " | ".join(query_parts)
                if query_parts
                else "일반적인 자동차를 추천해주세요"
            )

            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(f"**선택 조건:**\n\n{user_input}")

            with st.chat_message("assistant"):
                with st.spinner("추천 중입니다..."):
                    response = rag_chain.invoke(user_input)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
else:
    pass
