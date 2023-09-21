import numpy as np
import sympy as sp

from continuous_beam_solver.global_variables import *

# TODO se si fa in modo che Solver non dipenda piú da trave ma solo dalle variabili in essa, allora non è piu accoppiata (left_support, right, ncampate, ecc)


class Solver:
    def __init__(self, nCampate:int, left_support:str, right_support:str, spans_lenght:list[float], spans_ej:list[float]):
        self.nCampate = nCampate
        self.left_support = left_support
        self.right_support = right_support
        self.spans_lenght = spans_lenght
        self.spans_ej = spans_ej

    def generate_simbolic_variables(
        self,
    ):  # TODO mettere altre condizioni alle variabili?
        nCampate = self.nCampate
        L = sp.Matrix(
            nCampate,
            1,
            [
                sp.symbols(f"L_{i}", real="True", nonnegative="True")
                for i in range(1, nCampate + 1)
            ],
        )
        P = sp.Matrix(
            nCampate, 1, [sp.symbols(f"P_{i}") for i in range(1, nCampate + 1)]
        )
        Q = sp.Matrix(
            nCampate,
            1,
            [sp.symbols(f"Q_{i}", positive="True") for i in range(1, nCampate + 1)],
        )
        EJ = sp.Matrix(
            nCampate, 1, [sp.symbols(f"EJ_{i}") for i in range(1, nCampate + 1)]
        )
        return L, P, Q, EJ

    def generate_Flex_matrix(self) -> sp.Matrix:
        L, P, Q, EJ = self.generate_simbolic_variables()
        nCampate = self.nCampate

        # --- Principal Diagonal of the matrix ----
        flex_diag_sx = sp.Matrix(
            [sp.Rational(1, 3) * L[i] * EJ[i] ** -1 for i in range(0, nCampate)]
        )
        # Aggiunge uno zero all'inizio diventando n+1
        flex_diag_sx = sp.Matrix.vstack(sp.zeros(1, 1), flex_diag_sx)

        flex_diag_dx = sp.Matrix(
            [sp.Rational(1, 3) * L[i] * EJ[i] ** -1 for i in range(0, nCampate)]
        )
        # Aggiunge uno zero alla fine diventando n+1:
        flex_diag_dx = sp.Matrix.vstack(flex_diag_dx, sp.zeros(1, 1))

        flex_diag_tot = flex_diag_sx + flex_diag_dx

        # ---- Lower Diagonal of the matrix ----
        flex_lowerdiag = sp.Rational(1, 2) * sp.Matrix(flex_diag_sx[1 : nCampate + 1])

        # ---- Upper Diagonal of the matrix ----
        flex_upperdiag = sp.Rational(1, 2) * sp.Matrix(flex_diag_dx[0:-1])

        # ---- Total Flex matrix ----
        flex_gen = sp.zeros(nCampate + 1, nCampate + 1)
        for i in range(nCampate + 1):
            flex_gen[i, i] = flex_diag_tot[i]
        for i in range(nCampate):
            flex_gen[i, i + 1] = flex_upperdiag[i]
        for i in range(nCampate):
            flex_gen[i + 1, i] = flex_lowerdiag[i]

        return flex_gen

    def generate_P_vector_Q(self) -> sp.Matrix:
        L, P, Q, EJ = self.generate_simbolic_variables()
        nCampate = self.nCampate

        P_sx = sp.Matrix(
            [
                sp.Rational(1, 24) * Q[i] * L[i] ** 3 * EJ[i] ** -1
                for i in range(0, nCampate)
            ]
        )
        # Aggiunge uno zero all'inizio diventando n+1
        P_sx = sp.Matrix.vstack(sp.zeros(1, 1), P_sx)

        P_dx = sp.Matrix(
            [
                sp.Rational(1, 24) * Q[i] * L[i] ** 3 * EJ[i] ** -1
                for i in range(0, nCampate)
            ]
        )
        # Aggiunge uno zero alla fine diventando n+1:
        P_dx = sp.Matrix.vstack(P_dx, sp.zeros(1, 1))

        return P_sx + P_dx

    def generate_reduced_Flex_matrix_and_P_vector(self):
        nCampate = self.nCampate
        left_support = self.left_support
        right_support = self.right_support
        flex_gen = self.generate_Flex_matrix()
        P_gen = self.generate_P_vector_Q()

        # TODO aggiungere il Free support
        if left_support == "Simple" and right_support == "Simple":
            flex_rid = flex_gen[1:nCampate, 1:nCampate]
            P_rid = P_gen[1:nCampate]
        elif left_support == "Fixed" and right_support == "Simple":
            flex_rid = flex_gen[0:nCampate, 0:nCampate]
            P_rid = P_gen[0:nCampate]
        elif left_support == "Simple" and right_support == "Fixed":
            flex_rid = flex_gen[1 : nCampate + 1, 1 : nCampate + 1]
            P_rid = P_gen[1 : nCampate + 1]
        elif left_support == "Fixed" and right_support == "Fixed":
            flex_rid = sp.Matrix.copy(flex_gen)
            P_rid = sp.Matrix.copy(P_gen)

        flex_rid = sp.Matrix(flex_rid)
        P_rid = sp.Matrix(P_rid)
        return flex_rid, P_rid

    def generate_reduced_x_solutions(self) -> list[sp.Matrix]:
        nCampate = self.nCampate
        L, P, Q, EJ = self.generate_simbolic_variables()
        flex_rid, P_rid = self.generate_reduced_Flex_matrix_and_P_vector()
        # List of numeric values taken from Beam
        lenghts = self.spans_lenght
        ej = self.spans_ej

        # To solve the system for a generic Q=1 we have to subtistute values and then solve the systen for every span.
        # With Q values we have to subsistute them one at time and the must be other zero.
        #   Example with the first span: Q1 = 1, Q2 = 0 , Q3 = 0, ...
        #   Example with the second span: Q1 =0, Q2 = 1 , Q3 = 0, ...
        # To do this I'm using the Identidy matrix

        # Every x_solution_vector is added to a list
        list_of_reduced_x_solution_vectors = []
        for n_span in range(nCampate):
            flex_sub = (
                flex_rid.subs(zip(L, lenghts))
                .subs(zip(EJ, ej))
                .subs(zip(Q, np.array(sp.Identity(nCampate))[n_span]))
            )

            P_sub = (
                P_rid.subs(zip(L, lenghts))
                .subs(zip(EJ, ej))
                .subs(zip(Q, np.identity(nCampate)[n_span]))
            )

            # solve the system: # --- maybe there is a more efificient way
            x = -flex_sub.inv() * P_sub
            list_of_reduced_x_solution_vectors.append(x)
        return list_of_reduced_x_solution_vectors

    def generate_expanded_x_solutions(self) -> list[sp.Matrix]:
        """
        In base of boundary conditions return to initial lenghts the x_solution_vectors calculated in generate_reduced_x_solutions(self)
        """
        nCampate = self.nCampate
        left_support = self.left_support
        right_support = self.right_support
        list_of_reduced_x_solution_vectors = self.generate_reduced_x_solutions()

        # init with an identic list of the reduced one, and then add or not the zeros
        list_of_expanded_x_solution_vectors = list_of_reduced_x_solution_vectors  # TODO
        for n_span in range(nCampate):
            if left_support == "Simple" and right_support == "Simple":  # 0 prima e dopo
                # if there is only one span and no-incastre:
                # 'list_of_reduced_x_solution_vectors' fails to compute an adeguated matrix beacuse there aren't solutions of the system, then
                # 'generate_expanded_x_solutions' and 'generate_R_solutions' give an index error. So with this if i'm overwriting directly this case
                if self.nCampate > 1:
                    list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(
                        sp.zeros(1, 1), list_of_reduced_x_solution_vectors[n_span]
                    )
                    list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(
                        list_of_expanded_x_solution_vectors[n_span], sp.zeros(1, 1)
                    )
                else:
                    list_of_expanded_x_solution_vectors[n_span] = sp.zeros(2, 1)

            elif left_support == "Fixed" and right_support == "Simple":  # 0 dopo
                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(
                    list_of_reduced_x_solution_vectors[n_span], sp.zeros(1, 1)
                )

            elif left_support == "Simple" and right_support == "Fixed":  # 0 prima
                list_of_expanded_x_solution_vectors[n_span] = sp.Matrix.vstack(
                    sp.zeros(1, 1), list_of_reduced_x_solution_vectors[n_span]
                )

            elif left_support == "Fixed" and right_support == "Fixed":  # no 0
                pass

        return list_of_expanded_x_solution_vectors

    def generate_R_solutions(self, x: list[sp.Matrix]) -> list[sp.Matrix]:
        """
        R
        """
        nCampate = self.nCampate
        # x = self.generate_expanded_x_solutions()
        lenghts = self.spans_lenght

        list_of_R = []  # ---- oppure mettere - X + L/2:
        for n_span in range(nCampate):
            mat1 = sp.Matrix(
                [
                    (x[n_span][i + 1] - x[n_span][i]) / lenghts[i]
                    for i in range(0, nCampate)
                ]
            )
            mat2 = sp.Matrix(lenghts[n_span] / 2 * np.identity(nCampate)[n_span])
            list_of_R.append(mat1 + mat2)
        return list_of_R
    
    


