from continuous_beam_solver import Beam, Span
from continuous_beam_solver.internal_forces import BendingMoment, Shear
import matplotlib.pyplot as plt

c_1 = Span(lenght = 3.00, ej = 1000, q_max=10, q_min=2)
c_2 = Span(lenght = 4.50, ej = 1000, q_max=10, q_min=2)
c_3 = Span(lenght = 4.00, ej = 1000, q_max=10, q_min=2)
c_4 = Span(lenght = 5.00, ej = 1000, q_max=10, q_min=2)
c_5 = Span(lenght = 6.15, ej = 1000, q_max=10, q_min=2)
c_6 = Span(lenght = 4.00, ej = 1000, q_max=10, q_min=2)

trave = Beam(spans = [c_1, c_2, c_3, c_4, c_5, c_6], left_support="Simple", right_support="Fixed")

M = BendingMoment(trave)
M.plot_inviluppo()
plt.show()
