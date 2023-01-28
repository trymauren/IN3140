import numpy as np
from sympy import *

def settings(type): # Setter innstillinger for roboten
    if type == "joints" :           return 3
    if type == "input" :            return False
    if type == "cart_cord" :        return [0.00000,-323.90333,176.69884]
    if type == "joint_angles" :     return [3*pi/2,-pi/6,pi/4]
    s1,s2,s3 = symbols('theta_1,theta_2,theta_3')
    if type == "symbols" :          return [s1,s2,s3]
    if type == "joint_velocities" : return Matrix([0.1,0.05,0.05])

def main():

    joints = settings("joints")
    inp = settings("input")
    joint_angles = settings("joint_angles")
    cart_cord = settings("cart_cord")
    joint_velocities = settings("joint_velocities")
    symbols = settings("symbols")

    if(input("Vil du skrive inn egen DH- tabell? (trykk enter for å gjøre oblig2)") == "y"):
        inp = True

    print("\nTask 1, a)\n")
    FK = forward(create_DH("fk",joint_angles)) # må lage DH til forward, basert på vinklene satt øverst (denne er satt til RRR hvis ikke annen inputut gis)
    fk_print(FK,joint_angles)

    print("\nTask 1, b), c), and d)\n")
    IK,variableNames = inverse(create_DH("ik","not"),[round(FK[0],4),round(FK[1],4),round(FK[2],4)],3) # må lage DH til inverse (denne er satt til RRR hvis ikke annen input gis)
    ik_print(IK,variableNames,cart_cord)

    print("\nTask 2, a)\n")
    jacobianDH = create_DH("jb","not")
    jacobianFK = forward(jacobianDH)
    J = jacobian_whole(jacobianFK,jacobianDH)
    j_print(J)

    det = determinant_j(jacobian_upper(jacobianFK))
    det_print(det)

    sing = singularities(det)
    sing_print(sing)

    velocities = jacobian(joint_angles,joint_velocities)
    velocities_print(joint_angles,joint_velocities,velocities)

def create_DH(type,v): # lager forskjellige typer DH, denne er rotete...
    inp = settings("input")
    d1,d2,d3 = symbols('L_1,L_2,L_3')
    t1,t2,t3 = symbols('theta_1,theta_2,theta_3')
    a1,a2,a3 = symbols('alpha_1,alpha_2,alpha_3')
    r1,r2,r3 = symbols('r_1,r_2,r_3')

    DH_table = Matrix([ [t1, a1, r1, d1],
                        [t2, a2, r2, d2],
                        [t3, a3, r3, d3]])
    if(type == "jb"):
        return Matrix([[t1, -pi/2,      0,     d1],
                      [t2,      0,      r2,     0],
                      [t3,      0,      r3,     0]])

    if(inp):
        DH_table = dh_input(DH_table,input("Skriv inn antall joints"))
        DH_tableInverse = DH_table

    else:
        DH_table = Matrix([[v[0], -pi/2, 0, 100.9],
                           [v[1], 0, 222.1, 0],
                           [v[2], 0, 136.2, 0]])

        DH_tableInverse = Matrix([[t1, -pi/2,      0,     100.9],
                                 [t2,     0,      222.1,     0],
                                 [t3,     0,      136.2,     0]])

    if type == "ik":
        return DH_tableInverse

    return DH_table

def dh_input(DH,joints): # returnerer en DH- tabell gjennom bruker- input
    nameList = ["theta_","alpha_","r_","d_"]
    for name in nameList:
        for i in range(1,joints+1):
            print("Skriv inn " + name + str(i) + ":")
            nyN = input()
            if nyN != "":
                gammelN = name + str(i)
                DH = DH.subs(gammelN,nyN)

def A_matrix(variables): # returnerer generell A- matrise (kopiert fra gruppelærer)
    theta_i = variables[0]
    alpha_i = variables[1]
    r_i = variables[2]
    d_i = variables[3]

    theta, alpha, r, d  = symbols('theta alpha r d')

    rot_z = Matrix([[cos(theta), -sin(theta), 0, 0],
                    [sin(theta), cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    trans_z = Matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, d],
                      [0, 0, 0, 1]])

    trans_x = Matrix([[1, 0, 0, r],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    rot_x = Matrix([[1, 0, 0, 0],
                    [0, cos(alpha), -sin(alpha), 0],
                    [0, sin(alpha), cos(alpha), 0],
                    [0, 0, 0, 1]])

    A = rot_z @ trans_z @ trans_x @ rot_x

    A = A.subs([(theta, theta_i), (alpha, alpha_i), (r, r_i), (d, d_i) ])

    return A

def forward(DH): # returnerer FK til DH- tabell
    # DETTE BURDE GJØRES GENERELT!
    A1 = A_matrix([DH[0],DH[1],DH[2],DH[3]])
    A2 = A_matrix([DH[4],DH[5],DH[6],DH[7]])
    A3 = A_matrix([DH[8],DH[9],DH[10],DH[11]])
    A_matrices = [A1,A2,A3]
    FK = eye(4)
    for A in A_matrices: FK @= A
    return FK[0:3,3]

def jacobian_lower(DH,sysNumber): # returnerer nederste del av Jacobi
    if(sysNumber == 0):
        return [0,0,1]
    A1 = A_matrix([DH[0],DH[1],DH[2],DH[3]])
    A2 = A_matrix([DH[4],DH[5],DH[6],DH[7]])
    A3 = A_matrix([DH[8],DH[9],DH[10],DH[11]])
    A_matrices = [A1,A2,A3]
    FK = eye(4)
    count = 0
    for A in A_matrices:
        if(count == sysNumber):
            break
        FK @= A
        count += 1
    return FK[0:3,2]

def inverse(DH,point,joints): # returnerer IK
    print("\nBeregner invers kinematikk. Dette tar litt tid...")
    return inverse_all_solutions(forward(DH),point,settings("symbols"),joints),settings("symbols")

def inverse_all_solutions(FK,point,unknowns,joints): # returnerer alle svar på IK, invers kinematikk for (i utg. punktet) alle DH- tabeller
    unknowns = [symbols("theta_" + str(i)) for i in range(1,joints+1)]
    solutionAngles = solve((simplify(Eq(FK[0],point[0])),simplify(Eq(FK[1], point[1])),simplify(Eq(FK[2], point[2]))),(unknowns), rational = False, manual = True)
    return solutionAngles
    # return [n*180/np.pi % 360 for n in solutionAngles] #denne kan legges til hvis man vil ha ut vinklene mellom 0 og 360
def jacobian_whole(FK,DH): # returnerer hele Jacobi
    x,y,z = settings("symbols")
    f1,f2,f3 = FK

    i1,i2,i3 = jacobian_lower(DH,0)
    j1,j2,j3 = jacobian_lower(DH,1)
    k1,k2,k3 = jacobian_lower(DH,2)

    J = Matrix([[f1.diff(x),f1.diff(y),f1.diff(z)],
                [f2.diff(x),f2.diff(y),f2.diff(z)],
                [f3.diff(x),f3.diff(y),f3.diff(z)],
                [i1,j1,k1],
                [i2,j2,k2],
                [i3,j3,k3]])
    return J

def jacobian_upper(FK): # returnerer øvre del av Jacobi
    x,y,z = settings("symbols")
    f1,f2,f3 = FK
    J = Matrix([[f1.diff(x),f1.diff(y),f1.diff(z)],
                [f2.diff(x),f2.diff(y),f2.diff(z)],
                [f3.diff(x),f3.diff(y),f3.diff(z)]])
    return J

def determinant_j(J): # returnerer derminant til matrise
    return simplify(J.det())

def inverse_checker(IK,point): # sjekker at løsninger fra IK er riktige, gjennom å plugge de inn i FK
    fwik = forward(create_DH("fk",[IK[0],IK[1],IK[2]]))
    i = 0
    for p in point:
        if not round(p) == round(fwik[i]): return False
        i+=1
    return True

def jacobian(joint_angles,joint_velocities): # returnerer kartesisk hastighet for tupp
    t1,t2,t3 = settings("symbols")
    a1,a2,a3 = joint_angles
    DH = create_DH("fk",[t1,t2,t3])
    FK = forward(DH)
    J = jacobian_whole(FK,DH)
    Jinnsatt = J.evalf(subs={t1:a1,t2:a2,t3:a3}) @ joint_velocities

    return Jinnsatt[0:3]

def singularities(det): # returnerer singulærverdier basert på determinanten (ikke satt inn lengder)
    return solve(det)

def fk_print(FK,anglesFor): # printer
    names = ['x','y','z']
    print("\nKoordinatene vinklene " + str(anglesFor) + " gir gjennom FK- funksjonen er:\n")
    count = 0
    for koordinate in FK:
        print(names[count] + ": " + str(round(koordinate)))
        count += 1

def ik_print(IK,variableNames,cart_cord): # printer
    print("\nLøsninger på invers kinematikkproblemet til punktet " + str(cart_cord) + " : \n")
    solutionCount = 0
    for solution in IK:
        solution = list(solution)
        new = []
        count = 0
        for i in solution:
            newInside = []
            i = i*180/np.pi
            newInside.append(variableNames[count])
            newInside.append(round(i,4))
            new.append(newInside)
            count += 1

        print("Løsning " + str(solutionCount))
        pretty_print(new)
        print("Denne løsningen er: " + str(inverse_checker(IK[solutionCount],cart_cord)) +
        " (sjekket med foroverkinematikk- funksjonen i programmet)")
        print()
        solutionCount += 1

def j_print(J): # printer
    print("Jacobian- matrisen er gitt ved: \n")
    pretty_print(simplify(J))
    print()

def det_print(det): # printer
    print("\nDeterminanten til Jacobian- matrisen over er gitt ved: \n")
    pretty_print(simplify(det))
    print()

def sing_print(sing): # printer
    print("\nSingulærverdiene er:\n")
    pretty_print(sing)

def velocities_print(joint_angles,joint_velocities,velocities): # printer
    print("\n\nVinklene:")
    pretty_print(joint_angles)
    print()
    print("gir den kartesiske hastigheten:\n")
    printer = []
    for i in velocities:
        printer.append(str(i) + "rad/s")
    print(printer)

main()
