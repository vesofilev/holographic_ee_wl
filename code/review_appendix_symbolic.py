"""
Independent symbolic review of the proofs in Appendix A (metric degeneracy)
and the algebra of Appendix C (Wilson loop derivation) of draft.tex.

Fresh derivations -- deliberately not importing or reusing verify_appendix_a.py.

Checks:
 [A1] Step 1: r = z/sqrt(h) maps metric_general to Bilson form with
      g = alpha^2 f, chi = log(alpha^2 h), alpha = 1 - z h'/(2h)
      (all three metric components).
 [A2] eq (A.4): f'(0) = g'(0) + h'(0) for the reconstructed f = g(r(z))/alpha^2.
 [A3] Step 3: T = |f'(z_h)|/(4 pi) -- what does fixing T actually constrain?
      Compute f'(z_h) for the tilde metric symbolically.
 [A4] Step 4: alpha_GR = 1 - 3Qz/(4(1+Qz)); alpha_GR(1)|_{Q=1} = 5/8;
      positivity condition sqrt(1+z)(2+z/2) + eps z > 0 on [0,1];
      phi(z) = sqrt(1+z)(2+z/2)/z strictly decreasing => iff eps > -5 sqrt2/2.
 [A5] Step 5: generic UV series through O(r^6):
      g1 = 0; g2 = f2 + a^2/4 - 2 h2; g3 = f3 - 3 h3 (P3 = 0 exactly);
      for n = 2..6: dg_n/df_n = 1, dg_n/dh_n = -n, and
      P_n := g_n - f_n + n h_n depends only on lower-order coefficients.
 [C1] Nambu-Goto: -det G = (f h + z'^2)/z^4.
 [C2] H_W = -f h/(z^2 sqrt(f h + z'^2)); turning point value -sqrt(f* h*)/zhat*^2.
 [C3] z'^2 = f h (f h zhat*^4/(z^4 f* h*) - 1)  [integrand of eq (C.4)].
 [C4] V integrand equals (1/z^2)/sqrt(1 - z^4 f* h*/(zhat*^4 f h))  [eq (C.5)].
 [M1] Main-text eq (5.11): H = h^{3/2}/(z^2 sqrt(h + z'^2/f)) conserved form
      and z'^2 = f h (h^2 z*^4/(z^4 h*^2) - 1)  [integrand of eq (5.12)].
"""
import sympy as sp

z, r, Q, eps, zp = sp.symbols('z r Q epsilon z_prime', real=True)
a = sp.Symbol('a')
N = 7   # work modulo z^7  (verifies g_n through n = 6)

def trunc(expr, var, n=N):
    return sp.expand(sp.series(expr, var, 0, n).removeO())

print("=" * 72)
print("[A1] Step 1: Bilson coordinate map (generic h)")
print("=" * 72)
# generic h as symbolic function
hfun = sp.Function('h', positive=True)
ffun = sp.Function('f', positive=True)
hz, fz = hfun(z), ffun(z)
alpha = 1 - z*sp.diff(hz, z)/(2*hz)
rz = z/sp.sqrt(hz)
g_def = alpha**2 * fz
chi_def = sp.log(alpha**2 * hz)
# (i) spatial: h/z^2 dx^2 must equal 1/r^2 dx^2
sp_check = sp.simplify(hz/z**2 - 1/rz**2)
print("  g_xx:  h/z^2 - 1/r^2                        =", sp_check)
# (ii) radial: dz^2/(z^2 f) must equal dr^2/(r^2 g); dr = (alpha/sqrt(h)) dz
drdz = sp.simplify(sp.diff(rz, z))
rad_check = sp.simplify(1/(z**2*fz) - drdz**2/(rz**2*g_def))
print("  g_rr:  1/(z^2 f) - (dr/dz)^2/(r^2 g)        =", rad_check)
print("         (dr/dz = alpha/sqrt(h):", sp.simplify(drdz - alpha/sp.sqrt(hz)) == 0, ")")
# (iii) time: f/z^2 must equal g e^{-chi}/r^2
tt_check = sp.simplify(fz/z**2 - g_def*sp.exp(-chi_def)/rz**2)
print("  g_tt:  f/z^2 - g e^{-chi}/r^2               =", tt_check)

print()
print("=" * 72)
print("[A2] eq (A.4): tilde f'(0) = g'(0) + tilde h'(0)")
print("=" * 72)
# generic data g(r) = 1 + g1 r + g2 r^2 + g3 r^3, generic tilde h series
g1, g2s, g3s = sp.symbols('g1 g2 g3')
b1, b2, b3 = sp.symbols('b1 b2 b3')   # tilde h coefficients (b1 = h'(0))
ht = 1 + b1*z + b2*z**2 + b3*z**3
alt = trunc(1 - z*sp.diff(ht, z)/(2*ht), z, 4)
rt = trunc(z/sp.sqrt(ht), z, 4)
gfun = lambda x: 1 + g1*x + g2s*x**2 + g3s*x**3
ft = trunc(sp.expand(gfun(rt)) * trunc(1/alt**2, z, 4), z, 4)
fp0 = sp.simplify(sp.diff(ft, z).subs(z, 0))
print("  tilde f'(0) =", fp0, "   [claim: g1 + b1]  -> match:", sp.simplify(fp0 - (g1 + b1)) == 0)

print()
print("=" * 72)
print("[A3] Step 3: what fixing the temperature actually constrains")
print("=" * 72)
# tilde f'(z_h) with g(r_h) = 0:  f'(z_h) = g'(r_h) * r'(z_h) / alpha(z_h)^2
#                                        = g'(r_h) / (sqrt(h(z_h)) alpha(z_h))
hh, hph, gpr = sp.symbols('h_h hp_h gp_rh', positive=True)  # h(z_h), h'(z_h), g'(r_h)
zh = sp.Symbol('z_h', positive=True)
al_h = 1 - zh*hph/(2*hh)
# direct chain rule of f = g(r(z))/alpha(z)^2 at z_h, using g(r_h) = 0:
fp_h = gpr * (al_h/sp.sqrt(hh)) / al_h**2
print("  tilde f'(z_h) = g'(r_h)/(sqrt(h(z_h)) * alpha(z_h))")
print("               =", sp.simplify(fp_h))
print("  -> alpha(z_h) contains h'(z_h): fixing T (with h(z_h) fixed by s and")
print("     g'(r_h) fixed by the data) FIXES h'(z_h). The claim 'constrains")
print("     f'(z_h) but not h'(z_h)' is inaccurate as stated.")
print("  -> however: deformations with delta_h(z_h) = delta_h'(z_h) = 0 (e.g.")
print("     delta_h2 = z^2(z_h - z)^2) preserve both s and T -> family still infinite.")

print()
print("=" * 72)
print("[A4] Step 4: alpha_GR, 5/8 value, positivity iff-condition")
print("=" * 72)
h_gr = (1 + Q*z)**sp.Rational(3, 2)
al_gr = sp.simplify(1 - z*sp.diff(h_gr, z)/(2*h_gr))
claim = 1 - 3*Q*z/(4*(1 + Q*z))
print("  alpha_GR - [1 - 3Qz/(4(1+Qz))] =", sp.simplify(al_gr - claim))
print("  alpha_GR(z=1, Q=1) =", al_gr.subs({z: 1, Q: 1}), " (claim 5/8)")
d_al = sp.simplify(sp.diff(claim, z))
print("  d(alpha_GR)/dz =", d_al, " < 0 for Q,z > 0 -> monotone decreasing: OK")
# positivity expression for delta_h1 = z(1-z), Q = 1, z_h = 1
ht1 = (1 + z)**sp.Rational(3, 2) + eps*z*(1 - z)
expr = sp.simplify(2*ht1 - z*sp.diff(ht1, z))
target = sp.sqrt(1 + z)*(2 + z/2) + eps*z
print("  2h - z h' - [sqrt(1+z)(2+z/2) + eps z] =", sp.simplify(expr - target))
# phi(z) = sqrt(1+z)(2+z/2)/z strictly decreasing on (0,1]
phi = sp.sqrt(1 + z)*(2 + z/2)/z
phip = sp.simplify(sp.diff(phi, z))
# assemble sign: phi' = (1/sqrt(1+z)) * (1/4 - (z+2)/z^2)
sign_form = sp.simplify(phip - (sp.Rational(1, 4) - (z + 2)/z**2)/sp.sqrt(1 + z))
print("  phi'(z) - (1/4 - (z+2)/z^2)/sqrt(1+z) =", sign_form)
print("  on (0,1]: (z+2)/z^2 >= 3 > 1/4  ->  phi' < 0, min at z=1: phi(1) =",
      sp.simplify(phi.subs(z, 1)), "= 5*sqrt(2)/2: iff-condition OK")

print()
print("=" * 72)
print("[A5] Step 5: generic UV series through O(r^6)")
print("=" * 72)
fs = {n: sp.Symbol(f'f{n}') for n in range(2, N)}
hs = {n: sp.Symbol(f'h{n}') for n in range(2, N)}
f_ser = 1 + a*z + sum(fs[n]*z**n for n in range(2, N))
h_ser = 1 + a*z + sum(hs[n]*z**n for n in range(2, N))
alpha_s = trunc(1 - z*sp.diff(h_ser, z)/(2*h_ser), z)
g_of_z = trunc(alpha_s**2 * f_ser, z)
# invert r = z/sqrt(h): iterate z <- r*sqrt(h(z))
z_of_r = r
for _ in range(N):
    z_of_r = trunc(r*sp.sqrt(h_ser.subs(z, z_of_r)), r)
inv_res = trunc(z_of_r/sp.sqrt(h_ser.subs(z, z_of_r)) - r, r)
print("  inversion residual r(z(r)) - r =", inv_res)
g_of_r = trunc(g_of_z.subs(z, z_of_r), r)
gn = {n: sp.expand(g_of_r.coeff(r, n)) for n in range(N)}
print("  g0 =", gn[0], "  g1 =", gn[1])
print("  g2 =", gn[2], "   [claim f2 + a^2/4 - 2 h2]:",
      sp.simplify(gn[2] - (fs[2] + a**2/4 - 2*hs[2])) == 0)
print("  g3 =", gn[3], "   [claim f3 - 3 h3, P3 = 0]:",
      sp.simplify(gn[3] - (fs[3] - 3*hs[3])) == 0)
ok_all = True
for n in range(2, N):
    dfn = sp.simplify(sp.diff(gn[n], fs[n]))
    dhn = sp.simplify(sp.diff(gn[n], hs[n]))
    Pn = sp.expand(gn[n] - fs[n] + n*hs[n])
    higher = [s for m in range(n, N) for s in (fs[m], hs[m]) if Pn.has(s)]
    ok = (dfn == 1) and (dhn == -n) and not higher
    ok_all = ok_all and ok
    print(f"  n={n}:  dg/df_n = {dfn}, dg/dh_n = {dhn}, "
          f"P_n lower-order-only: {not higher}   -> {'OK' if ok else 'FAIL'}")
print("  eq (A.7) structure g_n = f_n - n h_n + P_n(lower):",
      "VERIFIED through O(r^6)" if ok_all else "FAILED")

print()
print("=" * 72)
print("[C1-C4] Wilson loop derivation (Appendix C)")
print("=" * 72)
f_, h_ = sp.symbols('f h', positive=True)
fst, hst, zst = sp.symbols('f_* h_* zhat_*', positive=True)
# C1: induced metric determinant
Gtt = -f_/z**2
Gxx = h_/z**2 + zp**2/(z**2*f_)
detG = sp.simplify(Gtt*Gxx)
print("  -det G - (f h + z'^2)/z^4 =", sp.simplify(-detG - (f_*h_ + zp**2)/z**4))
# C2: Hamiltonian
L_ng = sp.sqrt(f_*h_ + zp**2)/z**2
H_w = sp.simplify(zp*sp.diff(L_ng, zp) - L_ng)
print("  H_W - [-f h/(z^2 sqrt(f h + z'^2))] =",
      sp.simplify(H_w + f_*h_/(z**2*sp.sqrt(f_*h_ + zp**2))))
print("  turning point (z'=0):", sp.simplify(H_w.subs(zp, 0).subs({f_: fst, h_: hst, z: zst})),
      "  [claim -sqrt(f* h*)/zhat*^2]")
# C3: solve conservation for z'^2
zp2 = sp.solve(sp.Eq(f_*h_/(z**2*sp.sqrt(f_*h_ + zp**2)), sp.sqrt(fst*hst)/zst**2), zp**2)
zp2 = sp.simplify(zp2[0])
claim_c3 = f_*h_*(f_*h_*zst**4/(z**4*fst*hst) - 1)
print("  z'^2 - f h (f h zhat*^4/(z^4 f* h*) - 1) =", sp.simplify(zp2 - claim_c3))
# C4: V integrand = L_ng / |z'| must equal (1/z^2)/sqrt(1 - z^4 f* h*/(zhat*^4 f h))
integrand = sp.simplify((sp.sqrt(f_*h_ + zp2)/z**2)/sp.sqrt(zp2))
claim_c4 = (1/z**2)/sp.sqrt(1 - z**4*fst*hst/(zst**4*f_*h_))
diff_c4 = sp.simplify(integrand - claim_c4)
print("  V-integrand - eq(C.5) form =", diff_c4,
      "" if diff_c4 == 0 else "   <- check domain assumptions")

print()
print("=" * 72)
print("[M1] main text eq (5.11)-(5.12): RT Hamiltonian for the GR class")
print("=" * 72)
L_rt = sp.sqrt(h_)*sp.sqrt(h_ + zp**2/f_)/z**2
H_rt = sp.simplify(zp*sp.diff(L_rt, zp) - L_rt)
print("  H_RT - [-h^{3/2}/(z^2 sqrt(h + z'^2/f))] =",
      sp.simplify(H_rt + h_**sp.Rational(3, 2)/(z**2*sp.sqrt(h_ + zp**2/f_))))
zst2, hst2 = sp.symbols('z_* h_star', positive=True)
zp2_rt = sp.solve(sp.Eq(h_**sp.Rational(3, 2)/(z**2*sp.sqrt(h_ + zp**2/f_)),
                        hst2/zst2**2), zp**2)
zp2_rt = sp.simplify(zp2_rt[0])
claim_m1 = f_*h_*(h_**2*zst2**4/(z**4*hst2**2) - 1)
print("  z'^2 - f h (h^2 z*^4/(z^4 h*^2) - 1) =", sp.simplify(zp2_rt - claim_m1))

print()
print("ALL SYMBOLIC CHECKS COMPLETE")
