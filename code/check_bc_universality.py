"""
Check that every boundary condition encoded in the network ansaetze is
IDENTICAL to its pure-AdS counterpart (referee point 2).

The encodings impose:
  (E1) RT surface endpoint exponent:  z ~ (l/2 - x)^{1/d}
  (E2) string endpoint exponent:      z ~ (L/2 - x)^{1/3}
  (E3) metric boundary values:        f(0) = h(0) = 1
  (E4) horizon condition:             f(z_h) = 0

Check: compute the exact profiles' endpoint exponents by numerical
integration of the first-integral relation for
  - pure AdS (f = h = 1)  vs  the reconstructed backgrounds
    (AdS5-Schwarzschild d=4; Gubser-Rocha AdS4 d=3, Q=1),
and fit gamma = d log z / d log(l/2 - x) in the near-boundary window.
If gamma matches the pure-AdS value for the nontrivial backgrounds, the
encoded exponent is universal across the class and carries no information
about the interior functions being reconstructed.

Usage: python code/check_bc_universality.py
"""
import numpy as np
from scipy.integrate import quad

Q = 1.0


def f_gr(zv):
    U = (1 + (1 + 3 * Q) * zv + (1 + 3 * Q + 3 * Q ** 2) * zv ** 2) / (1 + Q * zv) ** 1.5
    return (1 - zv) * U


def h_gr(zv):
    return (1 + Q * zv) ** 1.5


def endpoint_exponent(dxdz, z_star, z_lo=1e-5, z_hi=1e-3, npts=12):
    """gamma from fit log z = gamma log(s) + c, s = l/2 - x(z) = int_0^z dx/dz'."""
    zs = np.geomspace(z_lo, z_hi, npts)
    s = np.array([quad(dxdz, 0, zv, limit=200)[0] for zv in zs])
    gamma = np.polyfit(np.log(s), np.log(zs), 1)[0]
    return gamma


print("(E1) RT surface endpoint exponent")
print("-" * 64)
# d = 4 (AdS5): z'^2 = f (z*^6/z^6 - 1)  [from H = z^-3/sqrt(1+z'^2/f)]
for name, fz in [("pure AdS5      (f=1)     ", lambda zv: 1.0),
                 ("AdS5-Schw      (f=1-z^4) ", lambda zv: 1 - zv ** 4)]:
    zst = 0.5
    dxdz = lambda zv: 1.0 / np.sqrt(fz(zv) * ((zst / zv) ** 6 - 1)) if zv > 0 else 0.0
    g = endpoint_exponent(dxdz, zst)
    print(f"  {name}: gamma = {g:.5f}   (1/d = {1/4:.5f})")

# d = 3 (AdS4 GR): z'^2 = f h (h^2 z*^4 / (z^4 h*^2) - 1)
for name, fz, hz in [("pure AdS4      (f=h=1)   ", lambda zv: 1.0, lambda zv: 1.0),
                     ("Gubser-Rocha   (Q=1)     ", f_gr, h_gr)]:
    zst = 0.5
    hst = hz(zst)
    dxdz = lambda zv: 1.0 / np.sqrt(fz(zv) * hz(zv) * ((hz(zv) ** 2 * zst ** 4) / (zv ** 4 * hst ** 2) - 1)) if zv > 0 else 0.0
    g = endpoint_exponent(dxdz, zst)
    print(f"  {name}: gamma = {g:.5f}   (1/d = {1/3:.5f})")

print()
print("(E2) Wilson-loop string endpoint exponent")
print("-" * 64)
# z'^2 = f h (f h zhat*^4/(z^4 f* h*) - 1)
for name, fz, hz in [("pure AdS4      (f=h=1)   ", lambda zv: 1.0, lambda zv: 1.0),
                     ("Gubser-Rocha   (Q=1)     ", f_gr, h_gr)]:
    zst = 0.4
    fst, hst = fz(zst), hz(zst)
    dxdz = lambda zv: 1.0 / np.sqrt(fz(zv) * hz(zv) * ((fz(zv) * hz(zv) * zst ** 4) / (zv ** 4 * fst * hst) - 1)) if zv > 0 else 0.0
    g = endpoint_exponent(dxdz, zst)
    print(f"  {name}: gamma = {g:.5f}   (universal 1/3 = {1/3:.5f})")

print()
print("(E3) metric boundary values: f(0) = h(0) = 1 is the pure-AdS value by")
print("     definition of the asymptotically-AdS class (identical for every")
print("     candidate geometry).")
print("(E4) f(z_h) = 0 defines the horizon: state data (temperature), not")
print("     interior dynamics.")
print()
print("Conclusion: the encoded exponents/values coincide with pure AdS for")
print("all backgrounds in the class -> they carry no information about the")
print("interior functions f, h being reconstructed. The only remaining")
print("structural choice, the shared slope f'(0) = h'(0), is DERIVED from")
print("the data in Appendix A (g'(0) = 0), not assumed.")
