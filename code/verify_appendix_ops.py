"""
Verification for the new appendix "Boundary asymptotics from the dual
operator content" (referee point 2).

Part A: fix the Einstein-Maxwell-dilaton action for the AdS4 Gubser-Rocha
        background used in the paper,
            S = int d^4x sqrt(-g) [ R - (k/2)(dphi)^2 - (1/4) e^{a_c phi} F^2
                                    + 6 cosh(phi) ],
        with metric ds^2 = (1/z^2)[-f dt^2 + dz^2/f + h(dx^2+dy^2)],
        f = (1-z)U(z), h = (1+Qz)^{3/2}, and ansatz
        phi = p ln(1+Qz), A_t = q (1-z)/(1+Qz).
        Determine (k, a_c, p, q) so that ALL field equations hold exactly.

Part B: linearized scalar equation -> m^2 = -2 -> Delta(Delta-3) = -2,
        i.e. Delta = 1 or 2: the powers z^{3-Delta} = z and z^{Delta} = z^2
        in the ansatz encode the dual operator dimension.

Part C: generic near-boundary expansion. Plug
        f = 1 + f1 z + f2 z^2 + f3 z^3, h = 1 + h1 z + h2 z^2 + h3 z^3,
        phi = p1 z + p2 z^2 + p3 z^3, A_t = mu + b1 z + b2 z^2
        into the field equations and solve order by order; report which
        coefficients are fixed and which remain free (boundary data).

Usage: python code/verify_appendix_ops.py
"""
import sympy as sp

z, Q = sp.symbols('z Q', positive=True)
k, a_c, p, q = sp.symbols('k a_c p q')

# ------------------------------------------------------------------
# geometry machinery for diagonal metric depending on z
# ------------------------------------------------------------------
t, x, y = sp.symbols('t x y')
coords = [t, z, x, y]


def einstein_tensor(gdiag):
    """Einstein tensor (lower indices) for diagonal metric gdiag (list of 4)."""
    g = sp.diag(*gdiag)
    ginv = sp.diag(*[1 / gi for gi in gdiag])
    n = 4
    Gamma = [[[0] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                expr = 0
                for d in range(n):
                    expr += ginv[a, d] * (sp.diff(g[d, b], coords[c])
                                          + sp.diff(g[d, c], coords[b])
                                          - sp.diff(g[b, c], coords[d]))
                Gamma[a][b][c] = sp.simplify(expr / 2)
    Ric = sp.zeros(n)
    for b in range(n):
        for c in range(n):
            expr = 0
            for a in range(n):
                expr += sp.diff(Gamma[a][b][c], coords[a]) - sp.diff(Gamma[a][b][a], coords[c])
                for d in range(n):
                    expr += Gamma[a][a][d] * Gamma[d][b][c] - Gamma[a][c][d] * Gamma[d][b][a]
            Ric[b, c] = sp.together(expr)
    Rs = sum(ginv[i, i] * Ric[i, i] for i in range(n))
    G = sp.zeros(n)
    for i in range(n):
        G[i, i] = sp.together(Ric[i, i] - Rs * g[i, i] / 2)
    return G, g, ginv


def field_equations(f, h, phi, At, kv, av):
    """Return [E_tt, E_zz, E_xx, scalar, maxwell] for the EMD system."""
    gdiag = [-f / z ** 2, 1 / (z ** 2 * f), h / z ** 2, h / z ** 2]
    G, g, ginv = einstein_tensor(gdiag)
    V = 6 * sp.cosh(phi)
    dphi2 = ginv[1, 1] * sp.diff(phi, z) ** 2
    F2 = -2 * z ** 4 * sp.diff(At, z) ** 2          # F_{munu}F^{munu}
    eqs = []
    for i in [0, 1, 2]:                              # tt, zz, xx
        Tphi = kv / 2 * ((sp.diff(phi, z) ** 2 if i == 1 else 0)
                         - g[i, i] * dphi2 / 2)
        if i == 0:
            FF = sp.diff(At, z) ** 2 * ginv[1, 1] * (-1)   # F_{t z}F_t{}^z = A'^2 g^{zz} * (-1)?? see below
            FF = sp.diff(At, z) ** 2 * ginv[1, 1]
            TF = sp.exp(av * phi) / 2 * (FF - g[i, i] * F2 / 4)
        elif i == 1:
            FF = sp.diff(At, z) ** 2 * ginv[0, 0]
            TF = sp.exp(av * phi) / 2 * (FF - g[i, i] * F2 / 4)
        else:
            TF = sp.exp(av * phi) / 2 * (0 - g[i, i] * F2 / 4)
        eqs.append(sp.together(G[i, i] - (Tphi + TF + g[i, i] * V / 2)))
    # scalar: k Box phi + dV/dphi - (a_c/4) e^{a_c phi} F^2 = 0
    sqrtg = h / z ** 4
    box = (1 / sqrtg) * sp.diff(sqrtg * (z ** 2 * f) * sp.diff(phi, z), z)
    eqs.append(sp.together(kv * box + 6 * sp.sinh(phi) - av / 4 * sp.exp(av * phi) * F2))
    # maxwell: d/dz( sqrtg e^{a phi} F^{zt} ) = 0, F^{zt} = z^4 A'
    eqs.append(sp.together(sp.diff(sqrtg * sp.exp(av * phi) * z ** 4 * sp.diff(At, z), z)))
    return eqs


print("=" * 72)
print("PART A: fix (k, a_c, p, q) so the exact GR background solves the EOM")
print("=" * 72)
U = (1 + (1 + 3 * Q) * z + (1 + 3 * Q + 3 * Q ** 2) * z ** 2) / (1 + Q * z) ** sp.Rational(3, 2)
f_ex = (1 - z) * U
h_ex = (1 + Q * z) ** sp.Rational(3, 2)
phi_ex = p * sp.log(1 + Q * z)
At_ex = q * (1 - z) / (1 + Q * z)

eqs = field_equations(f_ex, h_ex, phi_ex, At_ex, k, a_c)
# use two structurally different equations at a sample point to solve for params
cands = []
for kv in [1, 2, 3, sp.Rational(3, 2)]:
    for av in [1, sp.Rational(1, 2), sp.sqrt(3), 2]:
        for pv in [sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4), 1]:
            # solve q^2 from the tt equation at a sample (z, Q) and test others
            trial = [e.subs({k: kv, a_c: av, p: pv}) for e in eqs]
            e0 = sp.simplify(trial[0].subs({z: sp.Rational(1, 3), Q: 2}))
            sols = sp.solve(sp.Eq(e0, 0), q ** 2)
            if not sols:
                continue
            q2 = sols[0]
            ok = True
            for e in trial:
                val = sp.simplify(e.subs({z: sp.Rational(1, 5), Q: 3, q: sp.sqrt(q2.subs({z: sp.Rational(1, 3), Q: 2}))}))
                # q2 must be (z,Q)-independent for consistency; re-derive at the new point
            # cleaner: check q^2 consistency across sample points
            e0b = sp.simplify(trial[0].subs({z: sp.Rational(1, 5), Q: 3}))
            sols_b = sp.solve(sp.Eq(e0b, 0), q ** 2)
            if not sols_b:
                continue
            # require q^2 to depend only on Q -> compare functional forms
            q2a = sp.simplify(sols[0])
            q2b = sp.simplify(sols_b[0])
            cands.append((kv, av, pv, q2a, q2b))

for kv, av, pv, q2a, q2b in cands:
    # guess q^2 = 3 Q (1+Q)^3-ish forms: test full equations with symbolic q^2 = c
    qq = sp.symbols('qq', positive=True)
    trial = [e.subs({k: kv, a_c: av, p: pv, q: sp.sqrt(qq)}) for e in eqs]
    # solve qq(Q) from tt at generic z: collect
    e0 = sp.simplify(trial[0])
    sol_q = sp.solve(sp.Eq(e0.subs(z, sp.Rational(1, 3)), 0), qq)
    if not sol_q:
        continue
    qq_of_Q = sp.simplify(sol_q[0])
    resid = [sp.simplify(e.subs(qq, qq_of_Q)) for e in trial]
    if all(r == 0 for r in resid):
        print(f"  SOLUTION: k = {kv}, a_c = {av}, p = {pv}, q^2 = {qq_of_Q}")
        print("  all five field equations vanish identically.")
        K_FOUND, A_FOUND, P_FOUND, Q2_FOUND = kv, av, pv, qq_of_Q
        break
else:
    raise SystemExit("no consistent parameter set found -- check conventions")

print()
print("=" * 72)
print("PART B: linearized scalar equation -> operator dimension")
print("=" * 72)
# k Box phi + 6 sinh(phi) ~ k Box phi + 6 phi = 0  ->  Box phi = -(6/k) phi
m2 = -6 / sp.Integer(K_FOUND)
print(f"  m^2 = -6/k = {m2}")
D = sp.symbols('Delta')
sols = sp.solve(sp.Eq(D * (D - 3), m2), D)
print(f"  Delta(Delta-3) = m^2  ->  Delta = {sols}")

print()
print("=" * 72)
print("PART C: generic near-boundary expansion -- free vs fixed coefficients")
print("=" * 72)
f1, f2, f3, h1, h2, h3 = sp.symbols('f1 f2 f3 h1 h2 h3')
p1, p2, p3, mu, b1, b2 = sp.symbols('p1 p2 p3 mu b1 b2')
NO = 4
f_g = 1 + f1 * z + f2 * z ** 2 + f3 * z ** 3
h_g = 1 + h1 * z + h2 * z ** 2 + h3 * z ** 3
phi_g = p1 * z + p2 * z ** 2 + p3 * z ** 3
At_g = mu + b1 * z + b2 * z ** 2

eqs_g = field_equations(f_g, h_g, phi_g, At_g, sp.Integer(K_FOUND), A_FOUND)
labels = ['E_tt', 'E_zz', 'E_xx', 'scalar', 'maxwell']
print("order-by-order coefficient equations (after clearing z powers):")
system = []
for lab, e in zip(labels, eqs_g):
    e = sp.series(sp.expand(e * z ** 4), z, 0, NO).removeO().expand()
    for n in range(NO):
        cn = sp.expand(e.coeff(z, n))
        if cn != 0:
            system.append(cn)
            print(f"  [{lab}, z^{n}]  {sp.factor(cn)} = 0")

unknowns = [h1, f2, h2, f3, h3, p2, b2]
sol = sp.solve(system, unknowns, dict=True)
print()
print("solving for", unknowns, ":")
for s in sol:
    for key, val in s.items():
        print(f"   {key} = {sp.simplify(val)}")
    free = [v for v in [f1, p1, p3, mu, b1] if not any(v == kk for kk in s)]
    print("   free (boundary) data at this order:", free)
