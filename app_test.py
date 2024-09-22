import streamlit as st

# List of iframe links
iframes = [
    '<iframe src="https://ourworldindata.org/grapher/emissions-from-food?tab=map" loading="lazy" style="width: 100%; height: 600px; border: 0px none;" allow="web-share; clipboard-write"></iframe>'
]

# Display each iframe using streamlit's HTML rendering component
for iframe in iframes:
    st.components.v1.html(iframe, height=600)  # Adjust height according to your needs
