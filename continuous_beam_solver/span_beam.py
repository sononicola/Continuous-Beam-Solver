import numpy as np

def combinations_generic(
    q_max_list: list, q_min_list: list, nCampate: int
) -> list[list]:
    # for testing:
    # q_max_list = ["S1", "S2", "S3", "S4", "S5", "S6"]
    # q_min_list = ["F1", "F2", "F3", "F4", "F5", "F6"]
    # nCampate = 6

    # S S S S S S ...
    comb_0 = q_max_list
    # S F S F S F ...
    comb_1 = [q_max_list[i] if i % 2 == 0 else q_min_list[i] for i in range(nCampate)]
    # F S F S F S...
    comb_2 = [q_max_list[i] if i % 2 == 1 else q_min_list[i] for i in range(nCampate)]

    """
        S S F S F S...
        F S S F S F...
        S F S S F S...

        comb_j is a couple of [ S[j] , S[j+1] ] at j index. Then is added the left side and the right side to it. 
        if j = 1: ['F1', 'S2', 'S3', 'F4', 'S5', 'F6']
        if j = 2: ['S1', 'F2', 'S3', 'S4', 'F5', 'S6']
        """
    combs_SS: list[list] = []

    for j in range(0, nCampate - 1):
        comb_j = [q_max_list[j], q_max_list[j + 1]]
        if j % 2 == 0:
            comb_right = [
                q_max_list[i] if i % 2 == 1 else q_min_list[i]
                for i in range(j + 2, nCampate)
            ]
            comb_left = [
                q_max_list[i] if i % 2 == 0 else q_min_list[i] for i in range(0, j)
            ]
        else:
            comb_right = [
                q_max_list[i] if i % 2 == 0 else q_min_list[i]
                for i in range(j + 2, nCampate)
            ]
            comb_left = [
                q_max_list[i] if i % 2 == 1 else q_min_list[i] for i in range(0, j)
            ]

        comb_left.extend(comb_j)
        comb_left.extend(comb_right)
        combs_SS.append(comb_left)

    # Add all combinations in a list of lists:
    combs: list[list] = [comb_0, comb_1, comb_2]
    combs.extend(combs_SS)

    return combs


class Span:
    def __init__(
        self, lenght: float, ej: float, q_max: float = 0.0, q_min: float = 0.0
    ):
        self.lenght = lenght
        self.ej = ej
        self.q_max = q_max
        self.q_min = q_min

    def set_lenght(self):  # boh non servirà credo
        pass


class Beam:
    def __init__(self, spans: list[Span], left_support: str, right_support: str):
        """
        Avaiable left and right supports: "Simple", "Fixed". "Free" not implemented yet!"
        """
        self.spans = spans
        self.left_support = left_support
        self.right_support = right_support

    def get_spans(self):  # boh non servirà credo
        return self.spans

    def add_span(self, new_span: Span):
        """Add a single Span object to the spans list.
        Don't add a list of object like add_span([x1,x2]), but use add_list_of_spans([x1,x2]) instead!

        x1 = Span(...) \n
        beam = Beam([]) \n
        beam.add_span(x1)
        """
        self.spans.append(new_span)

    def add_list_of_spans(self, list_of_spans: list[Span]):
        """
        Add a list of object Span to the spans list.
        To add a single Span object use add_span(x1) insted!

        x1 = Span(...) \n
        x2 = Span(...) \n
        beam = Beam([]) \n
        beam.add_span([x1,x2])
        """
        self.spans.extend(list_of_spans)

    def spans_lenght(self) -> list:
        """Return a list with spans' lenghts"""
        return [span.lenght for span in self.spans]

    def spans_total_lenght(self) -> float:
        """Return the sum of spans lenght"""
        return np.sum(self.spans_lenght(), dtype=float)

    def spans_cum_lenght(self) -> list:
        """Return the cumulative sum of spans where the first element is 0"""
        cum_sum = np.cumsum(self.spans_lenght(), dtype=float)
        cum_sum = np.insert(cum_sum, 0, 0.0)  # Add a 0 at the beginning
        return cum_sum

    def spans_ej(self) -> list:
        """Return a list with spans' ej"""
        return [span.ej for span in self.spans]

    def spans_q_max(self) -> list:
        """Return a list with spans' q_max"""
        return [span.q_max for span in self.spans]

    def spans_q_min(self) -> list:
        """Return a list with spans' q_min"""
        return [span.q_min for span in self.spans]

    def _combinations_values(self) -> list[list[float]]:
        "Return the loads applied to each span, for each combination"
        return combinations_generic(
            q_max_list=self.spans_q_max(),
            q_min_list=self.spans_q_min(),
            nCampate=len(self.spans),
        )

    def _combinations_names(self) -> list[str]:
        """
        Return the combinations' names that are used.
        Example:
        [
        ["SSSS"],
        ["SFSF"],
        ["SSFS"],
        ["FSSF"],
        ["SFSS"]
        ]
        """
        combs: list[list[str]] = combinations_generic(
            q_max_list=["S" for span in range(len(self.spans))],
            q_min_list=["F" for span in range(len(self.spans))],
            nCampate=len(self.spans),
        )
        combination_name: list[str] = []
        for comb in combs:
            combination_name.append("".join(comb))
        return combination_name

    def combinations(self) -> dict[str, list[float]]:
        "Return a dict with combinations' names and the respective loads values applied to each span"
        return dict(zip(self._combinations_names(), self._combinations_values()))

