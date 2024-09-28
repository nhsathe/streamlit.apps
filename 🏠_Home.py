import streamlit as st

st.set_page_config(
    page_title="Nishank Sathe on Streamlit",
    page_icon="👋",
)

st.write("# Welcome to Nishank's projects on Streamlit! 👋")
st.page_link("pages/1_🏭_Facility_Location_Selection.py", label="Facility Location Selection", icon="🏭")



footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #000;
        text-align: center;
        padding: 10px;
    }
    .footer a {
        color: #000;
        text-decoration: none;
        margin: 0 15px;
    }
    .footer a:hover {
        color: #0073b1;  /* LinkedIn color on hover */
    }
    </style>

    <div class="footer">
        <a href="https://www.linkedin.com/in/nhsathe310" target="_blank">
            <i class="fa fa-linkedin"></i> 
        </a>
        <a href="https://github.com/nhsathe" target="_blank">
            <i class="fa fa-github"></i> 
        </a>
        <a href="mailto:nsathe@clemson.edu">
            <i class="fa fa-envelope"></i>
        </a>
    </div>

    <!-- Load Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
"""


st.markdown(footer, unsafe_allow_html=True)

