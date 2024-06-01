import streamlit as st
from streamlit_option_menu import option_menu
import home
import setting
import detection
import dashboard
import detectionchanmi


with st.sidebar:
    choice = option_menu("ASAPv5", ["HOME", "SETTING" ,"DETECTION", "DASHBOARD"],
    icons=['bi bi-house-fill','bi bi-gear-fill', 'bi bi-clipboard2-x-fill','bi bi-graph-up'],
                        menu_icon="bi bi-pin-angle-fill", default_index=0,
                        styles={
                            "container": {"padding": "5!important", "background-color": "#FFFFFF"},
                            "icon": {"color": "black", "font-size": "25px"},
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#8DBBD3"},
                        }
                        )
                        
if choice == "HOME":
    home.run_home()
elif choice == "SETTING":
    setting.run_setting()
elif choice == "DETECTION":
    #detection.run_detection()
    detectionchanmi.run_detection1()
elif choice == "DASHBOARD":
    dashboard.run_dashboard()