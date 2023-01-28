import sympy
from sympy import *

g = symbols('g');
L1,L2,L3 = symbols('L_1,L_2,L_3')
t1,t2,t3 = symbols('theta_1,theta_2,theta_3')
L1,L2,L3 = symbols('L1,L2,L3')
q1,q2,q3 = symbols('q_1,q_2,q_3')
qdot1,qdot2,qdot3 = symbols('qdot_1,qdot_2,qdot_3')
w1,w2,w3 = symbols('omega_1,omega_2,omega_3')
theta = symbols('theta')

i1x,i2x,i3x = symbols('I_1.x,I_2.x,I_3.x')
i1y,i2y,i3y = symbols('I_1.y,I_2.y,I_3.y')
i1z,i2z,i3z = symbols('I_1.z,I_2.z,I_3.z')

# v her er q- prikk- vektoren som består av q1-prikk, q2-prikk og q3-prikk
v = Matrix([[qdot1,qdot1,qdot3]])

m1 = 0.3833
m2 = 0.2724
m3 = 0.1406


J = Matrix([[-sin(t1) * (L2 * cos(t2) + L3 * cos(t2 + t3)), -cos(t1) * (L2 * sin(t2) + L2 * sin(t2 + t3)), -cos(t1) * (L3 * sin(t2 + t3))],
           [cos(t1) * (L2 * cos(t2) + L3 * cos(t2 + t3)), -sin(t1) * (L2 * sin(t2) + L2 * sin(t2 + t3)), -sin(t1) * (L3 * sin(t2 + t3))],
           [0, L2 * cos(t2) + L3 * cos(t2 + t3), L3 * cos(t2 + t3)],
           [0,  sin(t1),    sin(t1)],
           [0,  -cos(t1),   -cos(t1)],
           [1,  0,  0]])

J_v = Matrix([[J[0],J[1],J[2]],
              [J[3],J[4],J[5]],
              [J[6],J[7],J[8]]])

J_w = Matrix([[J[9],J[10],J[11]],
              [J[12],J[13],J[14]],
              [J[15],J[16],J[17]]])


rot_z = Matrix([[cos(theta), -sin(theta), 0],
                [sin(theta), cos(theta), 0],
                [0, 0, 1]])

rot_z2 = rot_z.subs([(theta, t2)])
rot_z3 = rot_z.subs([(theta, t3)])

Rm1 = eye(3)

Rm2 = Rm1 @ rot_z2

Rm3 = Rm2 @ rot_z3

I1 = Matrix([[i1x,0,0],
             [0,i1y,0],
             [0,0,i1z]])

I2 = I1

I3 = I1

def P():

    h1 = L1/2
    h2 = L1 + L2/2 * sin(t2)
    h3 = L1 + L2 * sin(t2) + L3/2 * sin(t2 - t3)

    gravity = Matrix([0,0,g])

    r_c1 = Matrix([0,0,h1])
    r_c2 = Matrix([L2/2 * cos(t2),0,h2])
    r_c3 = Matrix([L2 * cos(t2) + L3/2 * cos(t2 - t3),0,h3])

    p1 = m1 * gravity.dot(r_c1)
    p2 = m2 * gravity.dot(r_c2)
    p3 = m3 * gravity.dot(r_c3)

    P = p1 + p2 + p3

    return P

def D_q():

    J_v1 = J_v.evalf(subs={t2:0,t3:0,L2:0,L3:0})
    J_v2 = J_v.evalf(subs={t3:0,L3:0})
    J_v3 = J_v

    J_w1 = J_w.evalf(subs={t2:0,t3:0,L2:0,L3:0})
    J_w2 = J_w.evalf(subs={t3:0,L3:0})
    J_w3 = J_w

    v_1 = m1 * Transpose(J_v1) @ J_v1
    v_2 = m2 * Transpose(J_v2) @ J_v2
    v_3 = m3 * Transpose(J_v3) @ J_v3

    w_1 = Transpose(J_w1) @ Rm1 @ I1 @ Transpose(Rm1) @ J_w1
    w_2 = Transpose(J_w2) @ Rm2 @ I2 @ Transpose(Rm2) @ J_w2
    w_3 = Transpose(J_w3) @ Rm3 @ I3 @ Transpose(Rm3) @ J_w3

    k_1 = v_1 + w_1
    k_2 = v_2 + w_2
    k_3 = v_3 + w_3

    D_q = k_1 + k_2 + k_3

    return D_q

def K(D_q,v):
    return 1/2 * v * D_q * Transpose(v)

def euler(D_q,P):

    # a = første, b = andre, c = tredje del; av likning 6.65 i ny bok
    syms = symbols('theta_1,theta_2,theta_3')

    # Liste med Tau_1, Tau_2, Tau_3
    Tau_k = []

    for k in range(3):

        t_k = 0

        c = P.diff(syms[k])

        t_k += c

        for i in range(3):

            m = D_q.row(k)

            a = m[i]

            a *= (symbols('qddot' + str(i + 1)))

            t_k += a

            for j in range(3):

                f = m[j]

                b = 1/2 * ( f.diff(syms[i]) + f.diff(syms[j]) - f.diff(syms[k]) )

                b *= (symbols('qdot' + str(i + 1)) * (symbols('qdot' + str(j + 1))))

                t_k += b

        Tau_k.append(simplify((t_k)))

    return Tau_k


P = simplify(P())
print("\n\n\n------------------------------------------------------------------------------\n\n\n")
print("Potensiell energi:\n\n")
pretty_print(P)

D_q = D_q()
K = K(D_q,v)
print("\n\n\n------------------------------------------------------------------------------\n\n\n")
print("Kinetisk energi:")
print("..tar litt tid..\n\n")
pretty_print(simplify(K))

Euler_Lagrange = euler(D_q,P)
print("\n\n\n------------------------------------------------------------------------------\n\n\n")
print("Euler- Lagrange:\n\n")
print("..tar litt tid..\n\n")
pretty_print(Euler_Lagrange)
