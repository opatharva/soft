

import numpy as np



class FuzzySet:

    def __init__(self, membership_values):

        self.membership_values = np.array(membership_values)



    def union(self, other):

        return FuzzySet(np.maximum(self.membership_values, other.membership_values))



    def intersection(self, other):

        return FuzzySet(np.minimum(self.membership_values, other.membership_values))



    def complement(self):

        return FuzzySet(1 - self.membership_values)



    def __repr__(self):

        return f"FuzzySet({self.membership_values})"





if __name__ == "__main__":

    A = FuzzySet([0.2, 0.5, 0.8, 1.0])

    B = FuzzySet([0.1, 0.4, 0.6, 0.9])



    print(f"A: {A}\nB: {B}")

    print(f"Union: {A.union(B)}")

    print(f"Intersection: {A.intersection(B)}")

    print(f"Complement of A: {A.complement()}")

    print(f"Complement of B: {B.complement()}")


