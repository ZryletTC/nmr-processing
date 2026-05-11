import lmfit
from sympy import var
from sympy.parsing.mathematica import parse_mathematica

R1, R2, M1, M2, k, t, t0 = var('R1 R2 M1 M2 k t t0')

M1str = "1/(2 Sqrt[4 k^2+(R1-R2)^2])E^(1/2 (-2 k t-R1 t-Sqrt[4 k^2+(R1-R2)^2] t-R2 t+2 k t0+R1 t0-Sqrt[4 k^2+(R1-R2)^2] t0+R2 t0)) (E^(Sqrt[4 k^2+(R1-R2)^2] t0) (-2 k M2+M1 (R1+Sqrt[4 k^2+(R1-R2)^2]-R2))+E^(Sqrt[4 k^2+(R1-R2)^2] t) (2 k M2+M1 (-R1+Sqrt[4 k^2+(R1-R2)^2]+R2)))"

M2str = "1/(2 Sqrt[4 k^2+(R1-R2)^2])E^(1/2 (-2 k t-R1 t-Sqrt[4 k^2+(R1-R2)^2] t-R2 t+2 k t0+R1 t0-Sqrt[4 k^2+(R1-R2)^2] t0+R2 t0)) (E^(Sqrt[4 k^2+(R1-R2)^2] t) (2 k M1+M2 (R1+Sqrt[4 k^2+(R1-R2)^2]-R2))+E^(Sqrt[4 k^2+(R1-R2)^2] t0) (-2 k M1+M2 (-R1+Sqrt[4 k^2+(R1-R2)^2]+R2)))"

M1t = parse_mathematica(M1str)
M2t = parse_mathematica(M2str)
