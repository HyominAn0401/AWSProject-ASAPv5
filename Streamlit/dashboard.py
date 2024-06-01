import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px

def run_dashboard():
    st.title("📊 DASHBOARD")
    st.text('')
    
    # Load data
    df = pd.read_excel('data.xlsx')

    # Split 'dangerous' items into individual rows
    df = df.assign(dangerous=df['dangerous'].str.split('/')).explode('dangerous')
    df['dangerous'] = df['dangerous'].str.strip()  # Remove any leading/trailing whitespace
    
    # Total number of unique bags
    total_count = df['bag'].nunique()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('오늘 검사한 가방의 개수')
        st.write(f"<p style='font-size:60px; font-weight:bold;'>{total_count}</p>", unsafe_allow_html=True)

    with col2:
        st.subheader('오늘 탐지된 위해물품 개수')
        st.write(f"<p style='font-size:60px; font-weight:bold;'>4</p>", unsafe_allow_html=True)
    
    # Count occurrences of each dangerous item
    category_counts = df['dangerous'].value_counts()
    top_categories = category_counts.head(5)
    
    # Bar chart
    fig2 = px.bar(x=top_categories.index, y=top_categories.values, labels={'x': '위해물품', 'y': '갯수'}, title=' ', color_discrete_sequence=px.colors.qualitative.Set2)
    fig2.update_layout(width=350, height=400)  # Adjust graph size

    # Pie chart
    fig = px.pie(values=category_counts.values, names=category_counts.index, title=' ', color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(width=350, height=400)  # Adjust graph size

    # Two columns for charts
    col3, col4 = st.columns(2)

    with col3:
        st.subheader('많이 들어온 위해물품 TOP5')
        st.plotly_chart(fig2)

    with col4:
        st.subheader('탐지된 위해물품의 비율')
        st.plotly_chart(fig)

