import streamlit as st
from streamlit_option_menu import option_menu

def run_home():
    image_file = 'asapv5.png'  # 이미지 파일 경로
    st.image(image_file, width=200)
    st.header('공연장 위해물품 탐지 시스템')
    st.text('X-RAY를 통해 위해물품을 탐지하고 확인할 수 있는 서비스 입니다.')
    st.text(' ')
    st.subheader('⚙  SETTING  ')
    st.text('▶ [DEMO] 영상 업로드 ')
    st.text('▶ 위해물품 추가 설정')
    st.text(' ')
    st.subheader('📷 DETECTION ')
    st.text('▶ 탐지결과 확인')
    st.text('▶ 위해물품에 따른 대응 가이드')
    st.text(' ')
    st.subheader('📊 DASHBOARD ')
    st.text('▶ 오늘 검사한 가방의 개수')
    st.text('▶ 오늘 탐지된 위해물품 개수')
    st.text('▶ 많이 들어온 위해물품 TOP5')
    st.text('▶ 탐지된 위해물품의 비율')
