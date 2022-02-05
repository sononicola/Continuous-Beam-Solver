import numpy as np

class Span:
    def __init__(self, lenght: float, ej: float, q_max: float = 0., q_min: float = 0.):
        self.lenght = lenght
        self.ej = ej
        self.q_max = q_max
        self.q_min = q_min

class Beam: 
    def __init__(self, spans: list = [] , supports: str = "incastre"):
        self.spans = spans
        self.supports = supports # suddividere in  left and right
    
    def spans_lenght(self) -> list:
        """Return a list with spans' lenghts"""
        return [span.lenght for span in self.spans]

    def spans_total_lenght(self) -> float:
        """Return the sum of spans lenght"""
        return np.sum(self.spans_lenght, dtype=float) 
    
    def spans_cum_lenght(self) -> list:
        """Return the cumulative sum of spans lenghts"""
        return np.cumsum(self.spans_lenght, dtype=float)

    def spans_ej(self) -> list:
        """Return a list with spans' ej"""
        return [span.ej for span in self.spans]
    
    def spans_q_max(self) -> list:
        """Return a list with spans' q_max"""
        return [span.q_max for span in self.spans]
    
    def spans_q_min(self) -> list:
        """Return a list with spans' q_min"""
        return [span.q_min for span in self.spans]


x1 = Span(1,2)
x2 = Span(5,6)
x3 = Span(1.5, 7.6)
beam = Beam([x1,x2,x3])

for span in beam.spans: 
    print(span.ej)

print(beam.spans_lenght())
print(beam.spans_ej())
print(beam.spans_q_max())
print(beam.spans_q_min())

beam.spans_total_lenght


