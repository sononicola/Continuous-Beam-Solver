import streamlit as st
from continuous_beam_solver.solver import Solver
from continuous_beam_solver.span_beam import Span, Beam
from continuous_beam_solver.internal_forces import BendingMoment, Shear
from continuous_beam_solver.tables import Table
from continuous_beam_solver.global_variables import *

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd


# -- GENERAL PAGE SETUP --
st.set_page_config(
    page_title="Continuous beam solver",
    page_icon="👷‍♂️",
    initial_sidebar_state="collapsed",
    layout="wide",
)
# -- SIDEBAR --

# -- PAGE CONTENT --
st.title("Continuous beam solver - beta")
st.error(
    "At the moment there is an error on table results when right support is a simple support. They have to be zero but there is a problem with indexes.... fixing "
)
st.warning("💡 With lenght in meters and EJ in kN/m2, then M will be in .. and V in ..")

supports_list = ["Simple", "Fixed"]  # ["Free", "Simple", "Fixed"]
a1, a2, a3 = st.columns(3)
with a1:
    left_support = st.selectbox(
        label="Left support", options=supports_list, index=1, key="left_support"
    )
with a2:
    nSpan = int(st.number_input(label="n span", min_value=1, value=3))
with a3:
    right_support = st.selectbox(
        label="Right support", options=supports_list, index=1, key="right_support"
    )

with st.form("input"): 
    # Every column creates a list
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        # st.write("Lenghts")
        lenghts = [
            st.number_input(
                label=f"Lenght {i}",
                min_value=1.0,
                step=1.0,
                format="%.3f",
                key=f"lenght {i}",
            )
            for i in range(1, nSpan + 1)
        ]
    with c2:
        # st.write("EJs")
        ejs = [
            st.number_input(
                label=f"EJ {i}",
                value=983623.0,
                min_value=1.0,
                step=1.0,
                format="%.3f",
                key=f"ej {i}",
            )
            for i in range(1, nSpan + 1)
        ]
    with c3:
        # st.write("EJs")
        q_max = [
            st.number_input(
                label=f"Q_max {i}",
                value=1.0,
                # min_value = 1.,
                step=1.0,
                format="%.3f",
                key=f"q_max {i}",
            )
            for i in range(1, nSpan + 1)
        ]
    with c4:
        # st.write("EJs")
        q_min = [
            st.number_input(
                label=f"Q_min {i}",
                value=0.0,
                # min_value = 0.,
                step=1.0,
                format="%.3f",
                key=f"q_min {i}",
            )
            for i in range(1, nSpan + 1)
        ]
    run_button = st.form_submit_button("Run 🏗")     
# -- INIT OBJECTS --

# List of Span objects created starting from each list taken above:
spans = [Span(lenghts[i], ejs[i], q_max[i], q_min[i]) for i in range(nSpan)]

beam = Beam(spans=spans, left_support=left_support, right_support=right_support)

# Alternative method:
# beam = Beam([], left_support=left_support, right_support=right_support)
# beam.add_list_of_spans(spans)

df_inputs = pd.DataFrame(
    columns=[f"C{i}" for i in range(1, nSpan + 1)],
    data=[beam.spans_lenght(), beam.spans_ej(), beam.spans_q_max(), beam.spans_q_min()],
    index=["Lenghts", "EJs", "Q_Maxs", "Q_Mins"],
)
st.table(df_inputs)

# -- RUNNING PROGRAM --

def run(beam: Beam):
    sol = Solver(beam)
    x = sol.generate_expanded_x_solutions()
    r = sol.generate_R_solutions(x)
    st.latex(r"\textup{Flex}_{generic} = " + sp.latex(sol.generate_Flex_matrix()))
    st.latex(r"\textup{P}_{generic} = " + sp.latex(sol.generate_P_vector_Q()))
    st.latex(
        sp.latex(sol.generate_reduced_Flex_matrix_and_P_vector()[0])
        + r"\cdot \vec{X}_{rid} = "
        + sp.latex(sol.generate_reduced_Flex_matrix_and_P_vector()[1])
    )
    st.latex(r"\hookrightarrow \textup{X}_{expanded} = " + sp.latex(x))
    st.latex(r"\hookrightarrow \textup{R} = " + sp.latex(r))

    # -- CALCULATING BENDING MOMENT --
    M = BendingMoment(beam, x, r)
    V = Shear(beam, x, r)

    # storing x,y coordinates of inviluppo plot
    M_cords_x = M.s_func
    M_cords_y_pos, M_cords_y_neg = M.inviluppo()

    V_cords_x = V.s_func
    V_cords_y_pos, V_cords_y_neg = V.inviluppo()

    # PLOTTING
    # BENDING MOMENT
    st.pyplot(M.plot_inviluppo()[0])

    with st.expander("👉 Click to see plots where Q = 1 is applied in each span"):
        st.write("Bending Moment")
        for span in range(len(beam.spans)):
            st.pyplot(M.plot_span_Q_1(span)[0])
        st.pyplot(M.plot_beam_Q_1()[0])

    # table results for bending moment
    M_df_results = Table.create_dataframe(
        header=Table.make_header(len(beam.spans)),
        rows=Table.make_body(
            M_cords_x, M_cords_y_pos, M_cords_y_neg, beam.spans_cum_lenght()
        ),
        index=["s", "M_neg", "M_pos"],
    )
    st.table(M_df_results)
    st.warning(
        "If Bending Moment values aren't 0.0 when the support is Simple, it's a problem due to approximation!"
    )

    df_results_M_latex = M_df_results.style.to_latex(
        position="H", hrules=True, siunitx=True
    )
    st.download_button(
        "💾 Save results as a LaTeX table",
        data=df_results_M_latex,
        mime="text/latex",
        file_name="results_M_table.tex",
    )

    # SHEAR
    st.pyplot(V.plot_inviluppo()[0])

    with st.expander("👉 Click to see plots where Q = 1 is applied in each span"):
        st.write("Bending Voment")
        for span in range(len(beam.spans)):
            st.pyplot(V.plot_span_Q_1(span)[0])
        st.pyplot(V.plot_beam_Q_1()[0])

    # table results for bending moment
    V_df_results = Table.create_dataframe(
        header=Table.make_header(len(beam.spans)),
        rows=Table.make_body(
            V_cords_x, V_cords_y_pos, V_cords_y_neg, beam.spans_cum_lenght()
        ),
        index=["s", "V_pos", "V_neg"],
    )
    st.table(V_df_results)
    st.warning("If Shear values aren't")

    df_results_V_latex = V_df_results.style.to_latex(
        position="H", hrules=True, siunitx=True
    )
    st.download_button(
        "💾 Save results as a LaTeX table",
        data=df_results_V_latex,
        mime="text/latex",
        file_name="results_V_table.tex",
    )


if run_button:
    run(beam=beam)
# mime del pdf: application/pdf

