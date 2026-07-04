"""
Complete symbolic + numerical verification of Appendix A (metric degeneracy),
addressing referee point (4): remove all heuristic/unverified steps.

Strategy: fully symbolic where cheap; for the high-order UV structure the
non-target Taylor coefficients are specialized to random rationals while the
target-order pair (f_n, h_n) is kept symbolic. Since g_n is a polynomial in
the Taylor coefficients, its linear structure in (f_n, h_n) is verified
exactly by this specialization (repeated over independent random draws).

Checks:
  (1) Exactness of the Bilson map: g = alpha^2 f, e^chi = alpha^2 h (symbolic)
  (2) ftilde'(0) = g'(0) + htilde'(0) for a general family member
  (3) UV structure: g1 = 0; g2 = f2 + a^2/4 - 2 h2 (fully symbolic);
      g_n = f_n - n h_n + P_n(lower orders) for n = 3..6
      => kernel (flat direction) at order n: delta f_n = n delta h_n
      Also prints g3 fully symbolically.
  (4) Step-4 numerics: exact min of alpha_GR on [0,1] (Q=1) and the exact
      epsilon-range for alpha_tilde > 0 with delta h = z(1-z).

Usage: python code/verify_appendix_a.py
"""
import numpy as np
import sympy as sp
from fractions import Fraction
import random

rs, z, a, b = sp.symbols('r z a b')


def series_compose_g(f_coefs, h_coefs, nmax):
    """Given dicts {order: coeff} for f = 1 + a z + ..., h = 1 + a z + ...,
    return the list of coefficients g_n of g(r) = alpha^2 f expressed in the
    Bilson coordinate r = z/sqrt(h), up to order nmax (inclusive)."""
    f_ser = 1 + sum(c * z ** n for n, c in f_coefs.items())
    h_ser = 1 + sum(c * z ** n for n, c in h_coefs.items())
    alpha = 1 - z * sp.diff(h_ser, z) / (2 * h_ser)
    g_of_z = sp.series(alpha ** 2 * f_ser, z, 0, nmax + 1).removeO()
    r_of_z = sp.series(z / sp.sqrt(h_ser), z, 0, nmax + 2).removeO()
    # invert r(z) order by order: z = r + sum c_k r^k
    ck = sp.symbols(f'c2:{nmax + 1}')
    z_ans = rs + sum(ck[k - 2] * rs ** k for k in range(2, nmax + 1))
    expr = r_of_z.subs(z, z_ans)
    expr = sp.series(sp.expand(expr), rs, 0, nmax + 1).removeO().expand() - rs
    sol = {}
    for k in range(2, nmax + 1):
        e_k = expr.coeff(rs, k).subs(sol)
        sol[ck[k - 2]] = sp.expand(sp.solve(e_k, ck[k - 2])[0])
    z_of_r = z_ans.subs(sol)
    g_of_r = sp.series(sp.expand(g_of_z.subs(z, z_of_r)), rs, 0, nmax + 1)
    g_of_r = g_of_r.removeO().expand()
    return [sp.expand(g_of_r.coeff(rs, n)) for n in range(nmax + 1)]


print("=" * 70)
print("(1) Bilson map exactness: g = alpha^2 f,  e^chi = alpha^2 h")
print("=" * 70)
hf = sp.Function('h', positive=True)(z)
ff = sp.Function('f')(z)
r_full = z / sp.sqrt(hf)
alpha_full = 1 - z * sp.diff(hf, z) / (2 * hf)
dr_dz = sp.simplify(sp.diff(r_full, z))
print("dr/dz - alpha/sqrt(h)      =", sp.simplify(dr_dz - alpha_full / sp.sqrt(hf)))
g_full = sp.simplify((dr_dz ** 2) * z ** 2 / r_full ** 2 * ff)
print("g / (alpha^2 f)            =", sp.simplify(g_full / (alpha_full ** 2 * ff)))
echi = sp.simplify(g_full * z ** 2 / (r_full ** 2 * ff))
print("e^chi / (alpha^2 h)        =", sp.simplify(echi / (alpha_full ** 2 * hf)))

print()
print("=" * 70)
print("(2) ftilde'(0) = g'(0) + htilde'(0)  [g1, b symbolic; rest random]")
print("=" * 70)
random.seed(7)
g1 = sp.symbols('g1')
ok = True
for trial in range(3):
    gc = {1: g1, **{n: sp.Rational(Fraction(random.randint(-9, 9), random.randint(1, 7)))
                    for n in range(2, 6)}}
    hc = {1: b, **{n: sp.Rational(Fraction(random.randint(-9, 9), random.randint(1, 7)))
                   for n in range(2, 6)}}
    g_ser = 1 + sum(c * rs ** n for n, c in gc.items())
    ht = 1 + sum(c * z ** n for n, c in hc.items())
    alph = 1 - z * sp.diff(ht, z) / (2 * ht)
    ft = g_ser.subs(rs, z / sp.sqrt(ht)) / alph ** 2
    fp0 = sp.series(ft, z, 0, 2).removeO().expand().coeff(z, 1)
    diff = sp.simplify(fp0 - (g1 + b))
    ok &= (diff == 0)
    print(f"  trial {trial}: ftilde'(0) - (g1 + b) = {diff}")
print("  PASS" if ok else "  FAIL")

print()
print("=" * 70)
print("(3) UV structure of g(r)")
print("=" * 70)
# fully symbolic to order 3 (small): gives g1, g2, g3 exactly
f2, f3, h2, h3 = sp.symbols('f2 f3 h2 h3')
g_sym = series_compose_g({1: a, 2: f2, 3: f3}, {1: a, 2: h2, 3: h3}, 3)
print("g1 =", g_sym[1], "  (must vanish)")
print("g2 =", g_sym[2])
print("   g2 - (f2 + a**2/4 - 2*h2) =", sp.simplify(g_sym[2] - (f2 + a ** 2 / 4 - 2 * h2)))
print("g3 =", sp.factor(g_sym[3]))

# orders 4..6: keep (f_n, h_n) symbolic, randomize lower orders
for n in range(4, 7):
    fn, hn = sp.symbols(f'f{n} h{n}')
    results = []
    for trial in range(2):
        fc = {1: sp.Rational(Fraction(random.randint(-9, 9), random.randint(1, 7)))}
        hc = {1: fc[1]}  # shared a
        for k in range(2, n):
            fc[k] = sp.Rational(Fraction(random.randint(-9, 9), random.randint(1, 7)))
            hc[k] = sp.Rational(Fraction(random.randint(-9, 9), random.randint(1, 7)))
        fc[n], hc[n] = fn, hn
        gn = series_compose_g(fc, hc, n)[n]
        cf, ch = gn.coeff(fn), gn.coeff(hn)
        rest = sp.expand(gn - cf * fn - ch * hn)
        clean = fn not in rest.free_symbols and hn not in rest.free_symbols
        results.append((cf, ch, clean))
    print(f"g{n}: coeff(f{n}) = {results[0][0]}, coeff(h{n}) = {results[0][1]}, "
          f"linear+lower-order only: {all(r[2] for r in results)} "
          f"(consistent across draws: {results[0][:2] == results[1][:2]})")

print()
print("=> kernel at order n: delta f_n = n * delta h_n (flat direction),")
print("   n = 1 case is the shared boundary derivative a itself.")

print()
print("=" * 70)
print("(4) Step-4 numerics (Q=1, z_h=1)")
print("=" * 70)
Q = 1.0
zz = np.linspace(1e-9, 1.0, 200001)
alpha_gr = 1 - (3 * Q * zz) / (4 * (1 + Q * zz))
print(f"min alpha_GR on [0,1] = {alpha_gr.min():.6f} at z = {zz[alpha_gr.argmin()]:.4f}")
print(f"exact: alpha_GR(z_h) = 1 - 3Q/(4(1+Q)) = {1 - 3 * Q / (4 * (1 + Q))} = 5/8")
h_gr = (1 + Q * zz) ** 1.5
def min_pos(eps):
    ht = h_gr + eps * zz * (1 - zz)
    htp = 1.5 * (1 + Q * zz) ** 0.5 + eps * (1 - 2 * zz)
    return (2 * ht - zz * htp).min()
eps_crit = -np.sqrt(2) * 2.5
print(f"analytic condition: 2h - z h' + eps z > 0 <=> eps > -5*sqrt(2)/2 = {eps_crit:.4f}")
for eps in [-3.6, eps_crit + 1e-3, -0.8, 0.0, 0.8, 5.0, 50.0]:
    print(f"  eps = {eps:8.3f}: alpha_tilde > 0 on [0,1]? {min_pos(eps) > 0}")
