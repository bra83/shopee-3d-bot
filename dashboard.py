import google.generativeai as genai
import streamlit as st

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")
resp = model.generate_content("Responda apenas OK")
st.write(resp.text)

