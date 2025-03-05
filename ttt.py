from sympy import *
import numpy as np
a1 = np.radians(34)
a2 = np.radians(122)
Ff=88000
Fv, Rbx, Rby = symbols('Fv Rbx Rby')
print(solve([Eq(Fv*np.cos(a1)+Ff*np.cos(a2)+Rbx, 0), Eq(Fv*np.sin(a1)+Ff*np.sin(122)+Rby, 0), 
      Eq(-0.1*Fv*np.sin(a1)-0.82*Fv*np.cos(a1)+5.5*Ff*np.sin(a2)+4.1*Ff*np.cos(a2), 0)], [Fv, Rbx, Rby]))

