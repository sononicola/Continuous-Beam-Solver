import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from solver import Span, Beam, Compute


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
# beam.add_list_of_spans(spans)

# Solo per test dato che gli if sono ancora da sistemare :
beam  = Beam(spans, supports='incastre-right')  #TODO da capire se meglio all'inizio eaggiunto o subito qui

st.write(f"{beam.spans_lenght() = }")
st.write(f"{beam.spans_total_lenght() = }")
st.write(f"{beam.spans_cum_lenght() = }")
st.write(f"{beam.spans_ej() = }")
st.write(f"{beam.spans_q_max() = }")
st.write(f"{beam.spans_q_min() = }")


run = Compute(beam)

st.write(f"{run.generate_Flex_matrix() = }")
st.write(f"{run.generate_expanded_x_solutions() = }")
st.write(f"{run.generate_R_solutions() = }")

st.latex(sp.latex(run.generate_Flex_matrix()))
st.latex(sp.latex(run.generate_expanded_x_solutions()))
st.latex(sp.latex(run.generate_R_solutions()))

st.pyplot(run.plot_bending_moment_beam_Q())