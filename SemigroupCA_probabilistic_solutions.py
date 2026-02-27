#############################################################
## Distributed under Creative Commons Atribution Licence (BY)
##  https://creativecommons.org/licenses/by/4.0/
## Author: Henryk Fukś, February 2026
#############################################################
## This program produces graphs of probabilities of occurrences
## of symbols 0,1,2 from experiment and theoretical formula
## for any of the 18 ternary CA rules induced by semigroups.
## Just change the number below ("rule=number"), the numbers 1..18
## are the same  as in the paper "Ternary cellular automata
## induced by semigroups of order 3 are solvable".
## You can also change initial probabilities p0,p1 as needed.
############################################################
## The output consists of: 
## 1) three contunuous curves, these are plots of theoretical
## formulae for  P(0),P(1), P(2) vs. n
## 2) three plots of data points marked as circles,
## these correspond the numerical values of P(0),P(1), P(2)
## obtained by iterating the CA rule on array of length M
## for T time steps.
## M must be large, if it is too small datapoints may not
## lay on theoretical curves (because theoretical curves 
## correspond to infinite M).
############################################################
  
import numpy as np 
import matplotlib.pyplot as plt 
#from matplotlib.colors import ListedColormap
import math
import sys

############################################################
#### ENTER RULE NUMBER 1..18 HERE ##########################
rule=2
### NOTE: For rule=1 you must set probabilities p0,p1 to 1/3
############################################################


M=100000 #number of space sites
T=30 #number of iterations
 
x = [0]*M
# below we create initial condition, array of random integers
# p0 is the probability of zero, p1 prob. of 1, p2 prob. of 2
p0=0.433333
p1=0.233333
p2=1-p0-p1
x = np.random.choice([0, 1, 2], size=M, p=[p0, p1, p2])

if rule==1 and abs(p0-1/3)+abs(p1-1/3)+abs(p2-1/3)>0.0001:
    print ("For rule 1, probabilities p0,p1 should be set to 1/3")
    sys.exit(1)

###############################################################
##########definitions of loca function for rules     ##########
def f1(x0, x1, x2):   return (x1+x2)%3
def f2(x0, x1, x2):   return 1+(x2**2-x2-1)*(x1**2-x1-1)
def f3(x0, x1, x2):   return 2*x1*(-2+x1)*x2*(-2+x2)
def f4(x0, x1, x2):   return 1+(x2-1)*(x1-1)
def f5(x0, x1, x2):   return 2-(-3+x1+x2)*(x1*x2-x1-x2)
def f6(x0, x1, x2):   return 1+(x2**2-3*x2+1)*(x1**2-3*x1+1)
def f7(x0, x1, x2):   return 2       
def f8(x0, x1, x2):   return x1*(-2+x1)*x2*(-2+x2)
def f9(x0, x1, x2):   return (1/4)*x1*x2*(x2-3)*(x1-3)
def f10(x0, x1, x2):   return -x1*x2*(x1*x2-x1-x2)
def f11(x0, x1, x2):   return 1+(x1-1)**2          
def f12(x0, x1, x2):   return  -x1*x2*(-2+x1)
def f13(x0, x1, x2):   return max(x1,x2)
def f14(x0, x1, x2):   return (1/2)*x1*x2*(3*x1*x2-5*x1-5*x2+9)
def f15(x0, x1, x2):   return x1*(x1*x2**2-x1*x2-x2**2-2*x1+x2+4)/2       
def f16(x0, x1, x2):   return 1+(x2-1)**2*(x1-1)
def f17(x0, x1, x2):   return x1
def f18(x0, x1, x2):   return (1/2)*x1*(x1*x2-2*x1-x2+4)


###############################################################
#######Definitions of theoretical probabilities:
###############################################################    

# this is Gould's sequence, Python def. from OEIS A001316    
def G(k): return 1<<k.bit_count()
    
def P0r01(n):   return 1/3
def P1r01(n):   return 1/3
def P2r01(n):   return 1/3

def P0r02(n):   return 1/2-(1/2)*(1-2*p0-2*p1)**(G(n))     
def P1r02(n):   return 0   
def P2r02(n):   return (1/2)*(1-2*p0-2*p1)**(G(n)) + 1/2   

def P0r03(n):  
    if n==1: 
        return   1-p1**2 
    else: 
        return 1 
def P1r03(n):   return 0   
def P2r03(n):   
    if n==1: 
        return   p1**2 
    else: 
        return 0 

def P0r04(n):   return -(1/2)*(1-p1)**(n+1-G(n))*(1-2*p0-p1)**G(n)+(1/2)*(1-p1)**(n+1)   
def P1r04(n):   return 1-(1-p1)**(n+1)   
def P2r04(n):   return (1/2)*(1-p1)**(n+1-G(n))*(1-2*p0-p1)**(G(n)) + (1/2)*(1-p1)**(n+1)   

def P0r05(n):   return 1/2-(1/2)*(-1+2*p0)**G(n)   
def P1r05(n):   return p1**(n+1)   
def P2r05(n):   return 1/2+(1/2)*(-1+2*p0)**G(n)-p1**(n+1)   

def P0r06(n):   return 1/2-(1/2)*(2*p0-1)**(G(n))    
def P1r06(n):   return 0   
def P2r06(n):   return (1/2)*(2*p0-1)**(G(n)) + 1/2   

def P0r07(n):   return 0    
def P1r07(n):   return 0   
def P2r07(n):   return 1   

def P0r08(n):   return 1-p1**(n+1)    
def P1r08(n):   return p1**(n+1)   
def P2r08(n):   return 0   

def P0r09(n):   return 1-(1-p0)**(n+1)    
def P1r09(n):   return (1-p0)**(n+1)   
def P2r09(n):   return 0   

def P0r10(n):   return (2*p0+p1-2)*p1**n+1   
def P1r10(n):   return p1**(n+1)   
def P2r10(n):   return 2*(1-p1-p0)*p1**n   

def P0r11(n):   return 0   
def P1r11(n):   return p1   
def P2r11(n):   return 1-p1   

def P0r12(n):   return 1-p1**n*(1-p0)   
def P1r12(n):   return p1**(n+1)   
def P2r12(n):   return p1**n*(1-p1-p0)   

def P0r13(n):   return p0**(n+1)   
def P1r13(n):   return (p0+p1)**(n+1)- p0**(n+1)   
def P2r13(n):   return 1-(p0+p1)**(n+1)   

def P0r14(n):   return 1-p1**(n+1)-(1-p0-p1)**(n+1)   
def P1r14(n):   return p1**(n+1)   
def P2r14(n):   return (1-p0-p1)**(n+1)   

def P0r15(n):   return 1-p1-(1-p0-p1)**(n+1)   
def P1r15(n):   return p1   
def P2r15(n):   return (1-p0-p1)**(n+1)   

def P0r16(n):   return p0*(1-p1)**n   
def P1r16(n):   return 1-(1-p1)**(n+1)   
def P2r16(n):   return (1-p0-p1)*(1-p1)**n   

def P0r17(n):   return p0  
def P1r17(n):   return p1   
def P2r17(n):   return 1-p0-p1   

def P0r18(n):   return p0*(1-(1-p0-p1)**(n+1))/(p0+p1)   
def P1r18(n):   return p1*(1-(1-p0-p1)**(n+1))/(p0+p1)   
def P2r18(n):   return (1-p0-p1)**(n+1)   
##############################################################

locfunctions = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]  

P0functions=[P0r01, P0r02, P0r03, P0r04, P0r05, P0r06, P0r07, P0r08, P0r09, P0r10, P0r11, P0r12, P0r13, P0r14, P0r15, P0r16, P0r17, P0r18]
P1functions=[P1r01, P1r02, P1r03, P1r04, P1r05, P1r06, P1r07, P1r08, P1r09, P1r10, P1r11, P1r12, P1r13, P1r14, P1r15, P1r16, P1r17, P1r18]
P2functions=[P2r01, P2r02, P2r03, P2r04, P2r05, P2r06, P2r07, P2r08, P2r09, P2r10, P2r11, P2r12, P2r13, P2r14, P2r15, P2r16, P2r17, P2r18]

def f(rule, x0,x1,x2):
    return locfunctions[rule - 1](x0,x1,x2)

def P0th(rulenr, n):
    if (n==0):
        return p0
    else:
        return P0functions[rulenr - 1](n)
    
def P1th(rule,n):
    if (n==0):
        return p1
    else:
        return P1functions[rule - 1](n)
    
def P2th(rule,n):
    if (n==0):
        return p2
    else:    
        return P2functions[rule - 1](n)

# array where {0,1,2,..,T-1} is the range of first index (rows), 
# {0,1,2,..,M-1} is the range of the second index (columns)
# this construction guaranties that entries can be assigned independently
A = [[0 for _ in range(M)] for _ in range(T)] # actual iterations

# iterate CA
for i in range(0,M):
    A[0][i]=x[i]
for t in range(0,T-1):
    for i in range(0,M):
        A[t+1][i]=f(rule,A[t][(i-1+M)%M], A[t][i], A[t][(i+1)%M])

# now we count nr of 0,1,2 in each row
nr0=[]
nr1=[]
nr2=[]
for t in range(0,T):
  nr0.append(A[t].count(0))
  nr1.append(A[t].count(1))
  nr2.append(A[t].count(2))
tlist=np.arange(0, T)

# create arrays of "experimental" probabilities
P0ex=np.array(nr0)/M
P1ex=np.array(nr1)/M
P2ex=np.array(nr2)/M

# cretete arrays of theoretical  probabilities
P0 = [P0th(rule, t) for t in tlist]
P1 = [P1th(rule, t) for t in tlist]
P2 = [P2th(rule, t) for t in tlist]

# plot the probabilities
plt.plot(tlist, P0ex,"o")
plt.plot(tlist, P0,label=r'$P_n(0)$')

plt.plot(tlist, P1ex,"o")
plt.plot(tlist, P1,label=r'$P_n(1)$')

plt.plot(tlist, P2ex,"o")
plt.plot(tlist, P2,label=r'$P_n(2)$')

plt.xlabel(r'$n$')
plt.ylabel(r'$P_n$')
plt.legend()
plt.grid(True)
plt.show()

print("Rule number as in the paper:", rule)
    
