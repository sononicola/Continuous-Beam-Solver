import streamlit as st
from solver import Span, Beam
import numpy as np

# -- INIT --
beam = Beam([]) # every time some widget is clicked, streamlit re-start everything from here 

# -- GENERAL PAGE SETUP --
st.set_page_config(
     page_title = "Continuous beam solver",
     page_icon = "👷‍♂️",
     initial_sidebar_state = "collapsed",
     layout = "wide"
)
# -- SIDEBAR --

# -- PAGE CONTENT --
st.title("Continuous beam solver")
st.warning("💡 With lenght in meters and EJ in kN/m2, then M will be in .. and V in ..")

supports_list = ["free", "simple support" , "encastre"] # Cambiare i nomi con più decenti 
a1, a2, a3 = st.columns(3)
with a1:
    left_support = st.selectbox(
            label = "Left support", 
            options = supports_list, 
            index= 1,
            key = "left_support"            
    ) 
with a2:
    nSpan = int(st.number_input(
                    label = "n span", 
                    min_value=1, 
                    value=3
                    )
    )
with a3:
    right_support = st.selectbox(
            label = "Right support", 
            options = supports_list, 
            index= 1,
            key = "right_support"            
    ) 


# Every column creates a list 
c1, c2 = st.columns(2)
with c1:
    #st.write("Lenghts")
    lenghts = [
        st.number_input(
            label = f"Lenght {i}",
            min_value = 1.,
            step = 1.,
            format = "%.3f",
            key = f"lenght {i}"
            )
        for i in range(1,nSpan+1)
        ]
with c2:
    #st.write("EJs")
    ejs = [
        st.number_input(
            label = f"EJ {i}",
            value = 983623.,
            min_value = 1.,
            step = 1.,
            format = "%.3f",
            key = f"ej {i}"
            )
        for i in range(1,nSpan+1)
        ]

# List of Span objects created starting from each list taken above:
spans = [Span(lenghts[i], ejs[i]) for i in range(nSpan)]

# Add the spans list to the Beam object
beam.add_list_of_spans(spans)

st.write(f"{beam.spans_lenght() = }")
st.write(f"{beam.spans_total_lenght() = }")
st.write(f"{beam.spans_cum_lenght() = }")
st.write(f"{beam.spans_ej() = }")
st.write(f"{beam.spans_q_max() = }")
st.write(f"{beam.spans_q_min() = }")


a = 12.14785
st.write(f"{a = :.2f}")

st.slider("prova colore")
st.radio("s",["c","a"])
st.sidebar.write("prova")
st.sidebar.button("asa")

st.latex(**r"\alpha", **"s")