
import sympy as sp
from sympy import *
import numpy as np
t_1 = Function()
t1 = Function('t_1')()
t1,t2,t3 = symbols('theta_1,theta_2,theta_3');
L1,L2 = symbols('L_1,L_2');
q1 = Function('q_1')(t1)
pretty_print(q1)
pretty_print(q1.diff(t1))
pretty_print(q1.diff(t1).doit())
q2 = Function('q_2')(t1)
q3 = Function('q_3')(t1)
m_1,m_2,m_3 = symbols('m_1,m_2,m_3');
r_1,r_2,r_3 = symbols('r_1,r_2,r_3');
s1 = Function('s_1')(sin(t1))
s2 = Function('s_2')(sin(t2))
c1 = Function('c_1')(cos(t1))
c2 = Function('c_2')(cos(t2))

#
# A = Matrix([[cos(t1),0,-sin(t1),0],
#            [sin(t1),0,cos(t1),0],
#            [0,-1,0,L1],
#            [0,0,0,1]]);
#
# B = Matrix([[cos(t2),-sin(t2),0,L2*cos(t2)],
#            [sin(t2),cos(t2),0,L2*sin(t2)],
#            [0,0,1,0],
#            [0,0,0,1]]);
#
# C = A@B;
#
# pretty_print(C);
# w_1 = Matrix([0,0,q_1]);
# w_1t = Transpose(w_1);
# R = Matrix([[cos(t1),0,-sin(t1)],
#             [sin(t1),0,cos(t1)],
#             [0,-1,0]]);
# I = Matrix([[0,0,0],
#             [0,0,0],
#             [0,0,(m_1*((r_1)**2))/2]]);
# Rt = Matrix([[cos(t1),sin(t1),0],
#             [0,0,-1],
#             [-sin(t1),cos(t1),0]]);
#
#
# K_1 = w_1t@R@I@Rt@w_1
# pretty_print(w_1)
# pretty_print(w_1t)
# pretty_print(R)
# pretty_print(Rt)
# pretty_print(I)
# pretty_print(K_1)

x = [-s1*c2*L2*q1 -c1*s2*L2*q2, c1*c2*L2*q1 -s1*s2*L2*q2, c2*L2*q2]
y = [-s1*c2*L2*q1 -c1*s2*L2*q2, c1*c2*L2*q1 -s1*s2*L2*q2, c2*L2*q2]


y = np.transpose(y,axes=None)
c = np.dot(x,y)
c = expand(c)
pretty_print(c)
d = c.diff(t1)

pretty_print(d.doit())
