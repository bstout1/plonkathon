from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis


@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    setup: Setup
    program: Program
    pk: CommonPreprocessedInput
    A: Polynomial
    B: Polynomial
    C: Polynomial
    PI: Polynomial
    L0: Polynomial
    Z_H: Polynomial
    T1: Polynomial
    T2: Polynomial
    T3: Polynomial
    beta: Scalar
    gamma: Scalar
    zeta: Scalar
    fft_cofactor: Scalar
    alpha: Scalar

    a_eval: Scalar
    b_eval: Scalar
    c_eval: Scalar

    s1_eval: Scalar
    s2_eval: Scalar
    z_shift_eval: Scalar

    X: Polynomial
    v: Scalar

    def __init__(self, setup: Setup, program: Program):
        self.group_order = program.group_order
        self.setup = setup
        self.program = program
        self.pk = program.common_preprocessed_input()

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1)

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2)

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        setup = self.setup
        group_order = self.group_order

        if None not in witness:
            witness[None] = 0

        # Compute wire assignments for A, B, C, corresponding:
        # - A_values: witness[program.wires()[i].L]
        # - B_values: witness[program.wires()[i].R]
        # - C_values: witness[program.wires()[i].O]
        a_vals = [] # left wires
        b_vals = [] # right wires
        c_vals = [] # output wires

        for gate_wire in program.wires():
            a_vals.append(Scalar(witness[gate_wire.L]))
            b_vals.append(Scalar(witness[gate_wire.R]))
            c_vals.append(Scalar(witness[gate_wire.O]))
        
        while len(a_vals) < len(self.pk.QL.values):
            a_vals.append(Scalar(0))
        while len(b_vals) < len(self.pk.QR.values):
            b_vals.append(Scalar(0))
        while len(c_vals) < len(self.pk.QO.values):
            c_vals.append(Scalar(0))

        # Construct A, B, C Lagrange interpolation polynomials for
        # A_values, B_values, C_values

        self.A = Polynomial(a_vals, Basis.LAGRANGE)
        self.B = Polynomial(b_vals, Basis.LAGRANGE)
        self.C = Polynomial(c_vals, Basis.LAGRANGE)

        # Compute a_1, b_1, c_1 commitments to A, B, C polynomials
        a_1 = setup.commit(self.A)
        b_1 = setup.commit(self.B)
        c_1 = setup.commit(self.C)
        # Sanity check that witness fulfils gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        )

        # Return a_1, b_1, c_1
        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order
        setup = self.setup
        roots_of_unity = Scalar.roots_of_unity(group_order)
        F_values = [0 for _ in range(group_order)]
        G_values = [0 for _ in range(group_order)]
        # Using A, B, C, values, and pk.S1, pk.S2, pk.S3, compute
        # Z_values for permutation grand product polynomial Z
        #
        # Note the convenience function:
        #       self.rlc(val1, val2) = val_1 + self.beta * val_2 + gamma
        Z_values = [Scalar(0) for _ in range(group_order + 1)]
        Z_values[0] = Scalar(1)

        for i in range(group_order):
            F_values[i] = self.rlc(self.A.values[i], roots_of_unity[i])*self.rlc(self.B.values[i], 2*roots_of_unity[i])*self.rlc(self.C.values[i], 3*roots_of_unity[i])
            G_values[i] = self.rlc(self.A.values[i], self.pk.S1.values[i])*self.rlc(self.B.values[i], self.pk.S2.values[i])*self.rlc(self.C.values[i], self.pk.S3.values[i])
            Z_values[i+1] = Z_values[i]*Scalar(F_values[i]/G_values[i])

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0

        # Construct Z, Lagrange interpolation polynomial for Z_values
        Z = Polynomial(Z_values, Basis.LAGRANGE)
        self.Z = Z
        # Cpmpute z_1 commitment to Z polynomial
        z_1 = setup.commit(Z)
        # Return z_1
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order
        setup = self.setup


        # Compute the quotient polynomial

        # List of roots of unity at 4x fineness, i.e. the powers of µ
        # where µ^(4n) = 1
        expanded_roots_of_unity=Scalar.roots_of_unity(4*group_order)
        # Using self.fft_expand, move A, B, C into coset extended Lagrange basis
        exp_A = self.fft_expand(self.A)
        exp_B = self.fft_expand(self.B)
        exp_C = self.fft_expand(self.C)
        # Expand public inputs polynomial PI into coset extended Lagrange
        exp_PI = self.fft_expand(self.PI)
        # Expand selector polynomials pk.QL, pk.QR, pk.QM, pk.QO, pk.QC
        # into the coset extended Lagrange basis
        exp_QL = self.fft_expand(self.pk.QL)
        exp_QR = self.fft_expand(self.pk.QR)
        exp_QM = self.fft_expand(self.pk.QM)
        exp_QO = self.fft_expand(self.pk.QO)
        exp_QC = self.fft_expand(self.pk.QC)
        # Expand permutation grand product polynomial Z into coset extended
        # Lagrange basis
        exp_Z = self.fft_expand(self.Z)
        # Expand shifted Z(ω) into coset extended Lagrange basis
        exp_shifted_Z = exp_Z.shift(4)
        # Expand permutation polynomials pk.S1, pk.S2, pk.S3 into coset
        # extended Lagrange basis
        exp_S1 = self.fft_expand(self.pk.S1)
        exp_S2 = self.fft_expand(self.pk.S2)
        exp_S3 = self.fft_expand(self.pk.S3)
        # Compute Z_H = X^N - 1, also in evaluation form in the coset
        fft_cofactor = self.fft_cofactor
        Z_H = Polynomial([(((fft_cofactor*Scalar(x)) **group_order)-1 ) for x in expanded_roots_of_unity], Basis.LAGRANGE)
        self.Z_H = Z_H
        # Compute L0, the Lagrange basis polynomial that evaluates to 1 at x = 1 = ω^0
        # and 0 at other roots of unity
        L0 = Polynomial([Scalar(1)]+[Scalar(0) for _ in range(group_order - 1)], Basis.LAGRANGE)
        self.L0 = L0
        X = Polynomial(expanded_roots_of_unity, Basis.LAGRANGE)
        self.X = X
        two_X = Polynomial([Scalar(2)*x for x in expanded_roots_of_unity], Basis.LAGRANGE)
        three_X = Polynomial([Scalar(3)*x for x in expanded_roots_of_unity], Basis.LAGRANGE) 
        # Expand L0 into the coset extended Lagrange basis
        L0_big = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )

        # Compute the quotient polynomial (called T(x) in the paper)
        # It is only possible to construct this polynomial if the following
        # equations are true at all roots of unity {1, w ... w^(n-1)}:
        # 1. All gates are correct:
        #    A * QL + B * QR + A * B * QM + C * QO + PI + QC = 0
        #
        # 2. The permutation accumulator is valid:
        #    Z(wx) = Z(x) * (rlc of A, X, 1) * (rlc of B, 2X, 1) *
        #                   (rlc of C, 3X, 1) / (rlc of A, S1, 1) /
        #                   (rlc of B, S2, 1) / (rlc of C, S3, 1)
        #    rlc = random linear combination: term_1 + beta * term2 + gamma * term3
        #
        # 3. The permutation accumulator equals 1 at the start point
        #    (Z - 1) * L0 = 0
        #    L0 = Lagrange polynomial, equal at all roots of unity except 1
        QUOT_Big = ((exp_A*exp_B*exp_QM + exp_A*exp_QL + exp_B*exp_QR + exp_C*exp_QO + exp_PI + exp_QC) + ((self.rlc(exp_A, X* fft_cofactor)*self.rlc(exp_B, two_X*fft_cofactor)*self.rlc(exp_C, three_X*fft_cofactor)*exp_Z)*self.alpha) - ((self.rlc(exp_A, exp_S1)*self.rlc(exp_B, exp_S2)*self.rlc(exp_C,exp_S3)*exp_shifted_Z)*(self.alpha))+((exp_Z - Scalar(1))*L0_big*(self.alpha*self.alpha)))/Z_H
        # Sanity check: QUOT has degree < 3n
        assert (
            self.expanded_evals_to_coeffs(QUOT_Big).values[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Split up T into T1, T2 and T3 (needed because T has degree 3n - 4, so is
        # too big for the trusted setup)
        QUOT_coefficients = QUOT_Big.coset_extended_lagrange_to_coeffs(fft_cofactor)

        T1 = Polynomial(QUOT_coefficients.values[0:group_order], Basis.MONOMIAL).fft()
        T2 = Polynomial(QUOT_coefficients.values[group_order: 2*group_order], Basis.MONOMIAL).fft()
        T3 = Polynomial(QUOT_coefficients.values[2*group_order:3*group_order], Basis.MONOMIAL).fft()
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3
        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            T1.barycentric_eval(fft_cofactor)
            + T2.barycentric_eval(fft_cofactor) * fft_cofactor**group_order
            + T3.barycentric_eval(fft_cofactor) * fft_cofactor ** (group_order * 2)
        ) == QUOT_Big.values[0]

        print("Generated T1, T2, T3 polynomials")

        # Compute commitments t_lo_1, t_mid_1, t_hi_1 to T1, T2, T3 polynomials
        t_lo_1 = setup.commit(T1)
        t_mid_1 = setup.commit(T2)
        t_hi_1 = setup.commit(T3)

        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:
        # Compute evaluations to be used in constructing the linearization polynomial.

        # Compute a_eval = A(zeta), b_eval = B(zeta), c_eval = C(zeta)
        a_eval = self.A.barycentric_eval(self.zeta)
        self.a_eval = a_eval
        b_eval = self.B.barycentric_eval(self.zeta)
        self.b_eval = b_eval
        c_eval = self.C.barycentric_eval(self.zeta)
        self.c_eval = c_eval
        # Compute s1_eval = pk.S1(zeta), s2_eval = pk.S2(zeta)
        s1_eval = self.pk.S1.barycentric_eval(self.zeta)
        s2_eval = self.pk.S2.barycentric_eval(self.zeta)
        self.s1_eval = s1_eval
        self.s2_eval = s2_eval
        # Compute z_shifted_eval = Z(zeta * ω)
        z_shifted_eval = self.Z.barycentric_eval(self.zeta*Scalar.root_of_unity(self.group_order))
        self.z_shift_eval = z_shifted_eval
        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval)

    def round_5(self) -> Message5:
        # Evaluate the Lagrange basis polynomial L0 at zeta
        L0_eval = self.L0.barycentric_eval(self.zeta)

        # Evaluate the vanishing polynomial Z_H(X) = X^n - 1 at zeta
        Z_H_eval = self.zeta ** self.group_order-1
        PI_eval = self.PI.barycentric_eval(self.zeta)

        # Move T1, T2, T3 into the coset extended Lagrange basis
        exp_T1 = self.fft_expand(self.T1)
        exp_T2 = self.fft_expand(self.T2)
        exp_T3 = self.fft_expand(self.T3)

        # Move pk.QL, pk.QR, pk.QM, pk.QO, pk.QC into the coset extended Lagrange basis
        exp_QL = self.fft_expand(self.pk.QL)
        exp_QR = self.fft_expand(self.pk.QR)
        exp_QM = self.fft_expand(self.pk.QM)
        exp_QO = self.fft_expand(self.pk.QO)
        exp_QC = self.fft_expand(self.pk.QC)

        # Move Z into the coset extended Lagrange basis
        exp_Z = self.fft_expand(self.Z)

        # Move pk.S3 into the coset extended Lagrange basis
        exp_S3 = self.fft_expand(self.pk.S3)

        # Compute the "linearization polynomial" R. This is a clever way to avoid
        # needing to provide evaluations of _all_ the polynomials that we are
        # checking an equation betweeen: instead, we can "skip" the first
        # multiplicand in each term. The idea is that we construct a
        # polynomial which is constructed to equal 0 at Z only if the equations
        # that we are checking are correct, and which the verifier can reconstruct
        # the KZG commitment to, and we provide proofs to verify that it actually
        # equals 0 at Z
        #
        # In order for the verifier to be able to reconstruct the commitment to R,
        # it has to be "linear" in the proof items, hence why we can only use each
        # proof item once; any further multiplicands in each term need to be
        # replaced with their evaluations at Z, which do still need to be provided
        group_order = self.group_order
        c_eval = Polynomial([self.c_eval]*group_order*4, Basis.LAGRANGE)

        R = ((exp_QM*self.a_eval*self.b_eval+exp_QL*self.a_eval+exp_QR*self.b_eval+exp_QO*self.c_eval+PI_eval+exp_QC) +
            ((exp_Z*(self.rlc(self.a_eval, self.zeta)*self.rlc(self.b_eval, self.zeta*Scalar(2))*self.rlc(self.c_eval, self.zeta*Scalar(3))) - (self.rlc(c_eval, exp_S3)*self.rlc(self.b_eval, self.s2_eval) * self.rlc(self.a_eval, self.s1_eval))*self.z_shift_eval)*self.alpha) +
            (((exp_Z - Scalar(1))*L0_eval)*(self.alpha*self.alpha))-((exp_T1 + exp_T2*(self.zeta**self.group_order)+exp_T3*(self.zeta**(2*self.group_order)))*Z_H_eval))
        zeta=self.zeta
        R_coeffs = self.expanded_evals_to_coeffs(R).values
        R_actual = Polynomial(R_coeffs[0:group_order],Basis.MONOMIAL).fft()
        # Commit to R
        self.setup.commit(R_actual)
        # Sanity-check R
        assert R_actual.barycentric_eval(zeta) == 0

        print("Generated linearization polynomial R")

        # Generate proof that W(z) = 0 and that the provided evaluations of
        # A, B, C, S1, S2 are correct
        exp_A = self.fft_expand(self.A)
        exp_B = self.fft_expand(self.B)
        exp_C = self.fft_expand(self.C)
        exp_S1 = self.fft_expand(self.pk.S1)
        exp_S2 = self.fft_expand(self.pk.S2)
        # Move A, B, C into the coset extended Lagrange basis
        # Move pk.S1, pk.S2 into the coset extended Lagrange basis

        # In the COSET EXTENDED LAGRANGE BASIS,
        # Construct W_Z = (
        #     R
        #   + v * (A - a_eval)
        #   + v**2 * (B - b_eval)
        #   + v**3 * (C - c_eval)
        #   + v**4 * (S1 - s1_eval)
        #   + v**5 * (S2 - s2_eval)
        # ) / (X - zeta)
        W_Z = (R+((exp_A - self.a_eval)*self.v)+((exp_B-self.b_eval)*(self.v**2)) +((exp_C - self.c_eval)*(self.v**3))+((exp_S1 - self.s1_eval)*(self.v**4))+((exp_S2 - self.s2_eval)*(self.v**5))) / (self.X*self.fft_cofactor - self.zeta)

        W_z_coeffs = self.expanded_evals_to_coeffs(W_Z).values
        group_order = self.group_order

        # Check that degree of W_z is not greater than n
        assert W_z_coeffs[group_order:] == [0] * (group_order * 3)

        # Compute W_z_1 commitment to W_z
        W_z_1=self.setup.commit(W_Z)
        # Generate proof that the provided evaluation of Z(z*w) is correct. This
        # awkwardly different term is needed because the permutation accumulator
        # polynomial Z is the one place where we have to check between adjacent
        # coordinates, and not just within one coordinate.
        # In other words: Compute W_zw = (Z - z_shifted_eval) / (X - zeta * ω)
        W_zw = (exp_Z - self.z_shift_eval) / (self.X*self.fft_cofactor - (self.zeta * Scalar.root_of_unity(group_order)))
        W_zw_coeffs = self.expanded_evals_to_coeffs(W_zw).values

        # Check that degree of W_z is not greater than n
        assert W_zw_coeffs[group_order:] == [0] * (group_order * 3)

        # Compute W_z_1 commitment to W_z
        W_zw_1 = self.setup.commit(W_zw)
        print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma
