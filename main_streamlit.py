import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from solver import Span, Beam, Solver, BendingMoment
from tables import Table
import pandas as pd
from global_variables import *



# -- GENERAL PAGE SETUP --
st.set_page_config(
     page_title = "Continuous beam solver",
     page_icon = "👷‍♂️",
     initial_sidebar_state = "collapsed",
     layout = "wide"
)
# -- SIDEBAR --

# -- PAGE CONTENT --
st.title("Continuous beam solver - beta")
st.error("At the moment there is an error on table results when right support is a simple support. They have to be zero but there is a problem with indexes.... fixing ")
st.warning("💡 With lenght in meters and EJ in kN/m2, then M will be in .. and V in ..")

supports_list = ["Simple", "Fixed"] # ["Free", "Simple", "Fixed"] 
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
c1, c2, c3, c4 = st.columns(4)
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
with c3:
    #st.write("EJs")
    q_max = [
        st.number_input(
            label = f"Q_max {i}",
            value = 1.,
            min_value = 1.,
            step = 1.,
            format = "%.3f",
            key = f"q_max {i}"
            )
        for i in range(1,nSpan+1)
        ]
with c4:
    #st.write("EJs")
    q_min = [
        st.number_input(
            label = f"Q_min {i}",
            value = 0.,
            min_value = 0.,
            step = 1.,
            format = "%.3f",
            key = f"q_min {i}"
            )
        for i in range(1,nSpan+1)
        ]
# -- INIT OBJECTS --

# List of Span objects created starting from each list taken above:
spans = [Span(lenghts[i], ejs[i], q_max[i], q_min[i]) for i in range(nSpan)]

beam  = Beam(spans=spans, left_support=left_support, right_support=right_support) 

# Alternative method: 
# beam = Beam([], left_support=left_support, right_support=right_support)
# beam.add_list_of_spans(spans)

df_inputs = pd.DataFrame(
        columns=[f"C{i}" for i in range(1,nSpan+1)],
        data = [
            beam.spans_lenght(),
            beam.spans_ej(),
            beam.spans_q_max(),
            beam.spans_q_min()
        ],
        index = ["Lenghts", "EJs","Q_Maxs","Q_Mins"]
    )  
st.table(df_inputs)

# -- RUNNING PROGRAM --
run_button = st.button("Run 🏗")

def run(beam: Beam):
    sol = Solver(beam)
    x = sol.generate_expanded_x_solutions()
    r = sol.generate_R_solutions(x)
    st.latex(r"\textup{Flex} = " + sp.latex(sol.generate_Flex_matrix()))
    st.latex(r"\textup{P} = " + sp.latex(sol.generate_P_vector_Q()))
    st.latex(sp.latex(sol.generate_Flex_matrix()) + r"\cdot \vec{X} = " + sp.latex(sol.generate_P_vector_Q()))
    st.latex(r"\hookrightarrow \textup{X} = " + sp.latex(x))
    st.latex(r"\hookrightarrow \textup{R} = " + sp.latex(r))

# -- CALCULATING BENDING MOMENT --
    M = BendingMoment(beam, x, r)

    # storing x,y coordinates of inviluppo plot
    cords_x = M.s_func
    cords_y_pos, cords_y_neg = M.inviluppo()

    # plotting 
    with st.expander("👉 Click to see plots where Q = 1 is applied in each span"):
        for span in range(len(beam.spans)):
            st.pyplot(M.plot_span_Q_1(span))
        st.pyplot(M.plot_beam_Q_1())

    st.pyplot(M.plot_inviluppo())

    # table results for bending moment
    df_results_M = Table.create_dataframe(
        header=Table.make_header(len(beam.spans)),
        rows = Table.make_body(cords_x, cords_y_pos, cords_y_neg,beam.spans_cum_lenght()),
        index = ["s", "m_pos","m_neg"]
    )  
    st.table(df_results_M)
    st.warning("If Bending Moment values aren't 0.0 when the support is Simple, it's a problem due to approximation!")
    
    df_results_M_latex = df_results_M.style.to_latex(position='H', hrules=True, siunitx=True)
    st.download_button("💾 Save results as a LaTeX table", data=df_results_M_latex, mime="text/latex",file_name="results_M_table.tex")

if run_button:
    run(beam=beam)
#mime del pdf: application/pdf