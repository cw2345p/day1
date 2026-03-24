import streamlit as st
import pandas as pd
from kiwipiepy import Kiwi
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

# Kiwi 형태소 분석기
kiwi = Kiwi()

# --- 데이터 로딩 ---
@st.cache_data
def load_data(file_path):
    try:
        # 본문이 포함된 새 파일을 읽어옵니다.
        df = pd.read_csv(file_path)
        # 본문(content)이 비어있는 행은 제외합니다.
        df = df.dropna(subset=['query', 'content'])
        return df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류 발생: {e}")
        return None


# 뉴스 분석에 최적화된 불용어 리스트
korean_stopwords = {
    # 1. 일반적인 조사 및 의존 명사
    '것', '등', '및', '약', '또', '를', '수', '이', '그', '저', '더', '때', '중', '위', '뿐', 
    '즉', '한', '할', '감', '곳', '에서', '으로', '하는', '있는', '하고', '라고', '이다', '있다',
    
    # 2. 뉴스 기사 상투어 (매우 중요)
    '기자', '뉴스', '보도', '지난', '올해', '관련', '대한', '통해', '이번', '현재', '최근', 
    '오늘', '내년', '밝혔다', '전했다', '말했다', '강조했다', '덧붙였다', '설명했다', '나타났다',
    
    # 3. 언론사 정보 및 저작권 관련
    '특파원', '매체', '언론', '배포', '무단', '전재', '재배포', '금지', '저작권', 'ⓒ', 
    '연합뉴스', '뉴시스', '뉴스1', '전자신문', '주식회사', '제공', '사진', '출처', '이미지',
    
    # 4. 기타 분석 노이즈
    '사실', '경우', '때문', '정도', '직접', '일부', '모두', '가운데', '한편', '앞서', '관계자', 'nn', 'nnn', 'nnnn', 'nnnnn', 'nnnnnn', 'title', 'url', 'content'
}

# 데이터 파일명을 본문이 포함된 파일명으로 수정하세요.
df = load_data('news_data.csv')

if df is not None:
    # 사이드바: 검색어(query) 선택
    topics = sorted(df['query'].unique().tolist())
    selected_topic = st.sidebar.selectbox("📈 분석할 주제(Query) 선택", topics)

    st.title(f"🔍 '{selected_topic}' 뉴스 본문 분석")
    st.markdown("---")

    # --- 분석 로직 ---
    # 선택된 주제의 본문만 합치기
    topic_df = df[df['query'] == selected_topic]
    full_text = " ".join(topic_df['content'].astype(str).tolist())

    # 형태소 분석 및 전처리 함수
    def get_tokens(text):
        # 1. 특수문자 제거
        clean_text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
        # 2. 토큰화
        tokens = kiwi.tokenize(clean_text)
        # 3. 명사(NNG, NNP) 및 외국어(SL) 추출 + 1글자 및 불용어 제거
        # (korean_stopwords는 앞서 정의한 집합을 사용하세요)
        return [t.form for t in tokens if t.tag in {'NNG', 'NNP', 'SL'} 
                and t.form not in korean_stopwords and len(t.form) > 1]

    # 분석 실행
    with st.spinner('본문을 분석하여 키워드를 추출 중입니다...'):
        tokens = get_tokens(full_text)
        counts = Counter(tokens)
        top_10 = counts.most_common(10)

    # --- 시각화 영역 ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("☁️ 워드클라우드")
        if counts:
            wc = WordCloud(
                font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 윈도우는 파일명만 써도 기본적으로 인식합니다.
                width=800, height=450,
                background_color='white'
            ).generate_from_frequencies(counts)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("해당 주제에 분석할 텍스트 데이터가 부족합니다.")

    with col2:
        st.subheader("🔝 주요 키워드 TOP 10")
        if top_10:
            top_df = pd.DataFrame(top_10, columns=['키워드', '출현 빈도'])
            top_df.index = top_df.index + 1
            st.table(top_df)
            st.bar_chart(top_df.set_index('키워드'))
        else:
            st.write("데이터 없음")

    # (보너스) 선택한 주제의 기사 원문 리스트 보기
    with st.expander("📄 해당 주제의 기사 원문 보기"):
        for i, row in topic_df.iterrows():
            st.write(f"**[{row['title']}]**")
            st.caption(f"링크: {row['link']}")
            st.write(row['content'][:300] + "...") # 앞부분만 살짝 노출
            st.divider()