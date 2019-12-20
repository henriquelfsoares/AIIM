# AIIM
ARTICLE-RELATED CODE


#MODELO COM ENTRADA DE DADOS PELO USUÁRIO



import math
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import linalg as LA
get_ipython().magic('matplotlib inline')

t_inicial = time.time()
#definições de parâmetros

lamb = 182                                       #taxa de incrementação do estoque associada a oferta por doações
mi = 258                                         #taxa de esvaziamento do estoque associada a demanda dos hospitais
c = 70                                           #capacidade esperada de cada ônibus
estados = 1000                                   #número de estados, depois mudar
bus = 140                                        #custo do envio de um ônibus 
prod = 50                                        #custo da produção de uma bolsa de sangue
cf = 1000                                        #relativo ao custo de falta associado ao estado 0 (sem estoque) 
df = 80                                          #relativo ao decaimento exponencial do custo de falta de uma bolsa de sangue
ce1 = 0.7                                        #relativo ao crescimento linear do custo de estoque das bolsas de sangue 
ce2 = 0.1                                        #relativo ao ganho em economia de escala nos custos de estoque
ce3 = 0.7                                        #relativo a perca dos ganhos em economia de escala nos custos de estoque
n0 = 0                                           #variável de substituição referente aos estados negativos
n_bus = 2



#parêmetros escolhidos pelo usuário

lamb = input("input lambda:  ")
mi = input("input mi:  ")
c = input("input delta_lambda:  ")
estados = input("input s:  ")
bus = input("input cc:  ")
prod = input("input cp:  ")
cf = input("input c1:  ")
df = input("input d1:  ")
ce1 = input("input s1:  ")
ce2 = input("input s2:  ")
ce3 = input("input s3:  ")

lamb = int(lamb)
mi = int(mi)
c = int(c)
estados = int(estados)
bus = int(bus)
prod = int(prod)
cf = int(cf)
df = int(df)
ce1 = float(ce1)
ce2 = float(ce2)
ce3 = float(ce3)




#Inicialização do vetor de Custos

C = [0]* (estados)                               #Co,C1,C2,...,Cn-1,Cn -> Vetor de custos
Cf = [0]* (estados)                              #Vetor de custos de falta
Ce = [0]* (estados)                              #Vetor de custos de estoque
for i in range(estados):
    C[i] += cf*math.exp(-(i/df)) + ce1*i
#    Cf[i] += cf*math.exp(-(i/df))
#for i in range (0,round(estados/4)):
#    Ce[i] += ce1*i
#for i in range (round(estados/4),round(3*estados/4)):
#    Ce[i] += (Ce[round(estados/4)-1] + ce2*(i-round(estados/4)+1))
#for i in range (round(3*estados/4),estados):
#    Ce[i] += (Ce[round(3*estados/4)-1]+ ce3*(i-round(3*estados/4)+1))
#    
#for i in range(estados):
#    C[i] += Cf[i] + Ce[i]


v_old = [0] * len(C)
v_new = [0] * len(C)
v_new_menos_old = [0] * len(C)

mem_acoes = ['0bus'] * len(C)
gap = 100
it = 0

lista_estados = []
for i in range(1,estados-1):
    lista_estados.append(i)


       

v_p_sup = [0]*(n_bus+1)
v_p_inf = [0]*(n_bus+1)
v_custo_acoes = [0]*(n_bus+1)

for i in range(n_bus+1):
    v_p_sup[i] = round((lamb+i*c)/(lamb+i*c+mi),3)
    v_p_inf[i] = round(mi/(lamb+i*c+mi),3)
    v_custo_acoes[i] = v_p_sup[i]*(C[i+1]+v_old[i+1]+prod) + v_p_inf[0]*(C[i-1]+v_old[i-1])  

    
p_sup_0bus = round(lamb/(lamb+mi),3)                      #sup refere-se à probablidade de ir de um estado n para um n+1 
p_inf_0bus = round(mi/(lamb+mi),3)                        #inf refere-se à probabilidade de ir de um estado n+1 para um n 
p_sup_1bus = round((lamb+c)/(lamb+c+mi),3)
p_inf_1bus = round(mi/(lamb+c+mi),3)
p_sup_2bus = round((lamb+2*c)/(lamb+2*c+mi),3)
p_inf_2bus = round(mi/(lamb+2*c+mi),3)    
    
#-------------------------------------------------------------------------------------------------------------
#Algoritmo

#while gap > 0.1:
for i in range(10000):
    
    it += 1
    for j in range(len(v_old)):
        v_old[j] = v_new[j]
    v_new[0] = min(p_sup_0bus*(C[1]+v_old[0]) + p_inf_0bus*(C[0]+v_old[0]),\
                   p_sup_1bus*(C[1]+bus+v_old[0]) + p_inf_1bus*(C[0]+bus+v_old[0]), \
                   p_sup_2bus*(C[1]+2*bus+v_old[0]) + p_inf_2bus*(C[0]+2*bus+v_old[0]))
    if v_new[0] == p_sup_0bus*(C[1]+v_old[0]) + p_inf_0bus*(C[0]+v_old[0]):
        mem_acoes[0] = '0bus'
    elif v_new[0] == p_sup_1bus*(C[1]+bus+v_old[0]) + p_inf_1bus*(C[0]+bus+v_old[0]):
        mem_acoes[0] = '1bus'
    else:
        mem_acoes[0] = '2bus'

    for i in lista_estados:
        v_new[i] = min(p_sup_0bus*(C[i+1]+v_old[i+1]) + p_inf_0bus*(C[i-1]+v_old[i-1]), \
                       p_sup_1bus*(C[i+1]+bus+v_old[i+1]) + p_inf_1bus*(C[i-1]+bus+v_old[i-1]),  \
                       p_sup_2bus*(C[i+1]+2*bus+v_old[i+1]) + p_inf_2bus*(C[i-1]+2*bus+v_old[i-1]))
        if v_new[i] == p_sup_0bus*(C[i+1]+v_old[i+1]) + p_inf_0bus*(C[i-1]+v_old[i-1]):
            mem_acoes[5] = '0bus'
        elif v_new[i] == p_sup_1bus*(C[i+1]+bus+v_old[i+1]) + p_inf_1bus*(C[i-1]+bus+v_old[i-1]):
            mem_acoes[i] = '1bus'
        else:
            mem_acoes[i] = '2bus'

    v_new[estados-1] = min(p_sup_0bus*(C[estados-1]+v_old[estados-1]) + p_inf_0bus*(C[estados-2]+v_old[estados-estados-2]), \
                           p_sup_1bus*(C[estados-1]+bus+v_old[estados-1]) + p_inf_1bus*(C[estados-2]+bus+v_old[estados-2]),   \
                           p_sup_2bus*(C[estados-1]+2*bus+v_old[estados-1]) + p_inf_2bus*(C[estados-2]+2*bus+v_old[estados-2]))
    if v_new[estados-1] == p_sup_0bus*(C[estados-1]+v_old[estados-1]) + p_inf_0bus*(C[estados-2]+v_old[estados-2]):
        mem_acoes[estados-1] = '0bus'
    elif v_new[estados-1] == p_sup_1bus*(C[estados-1]+bus+v_old[estados-1]) + p_inf_1bus*(C[estados-2]+bus+v_old[estados-2]):
        mem_acoes[estados-1] = '1bus'
    else:
        mem_acoes[estados-1] = '2bus'


    for i in range(len(v_old)):
        v_new_menos_old[i] = v_new[i] - v_old[i]
    gap = max(v_new_menos_old) - min(v_new_menos_old)
    
t_final = time.time()    




#Criando a matriz de probabilidades
P = np.zeros((estados,estados))
for i in range(1,estados-1):
    for j in [i-1]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_inf[0]
        else:
            P[i][j] = v_p_inf[1]
    for j in [i+1]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_sup[0]
        else:
            P[i][j] = v_p_sup[1]
for i in [0]:
    for j in [i]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_inf[0]
        else:
            P[i][j] = v_p_inf[1]
    for j in [i+1]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_sup[0]
        else:
            P[i][j] = v_p_sup[1]
for i in [estados-1]:
    for j in [i-1]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_inf[0]
        else:
            P[i][j] = v_p_inf[1]
    for j in [i]:
        if mem_acoes[i] == '0bus':
            P[i][j] = v_p_sup[0]
        else:
            P[i][j] = v_p_sup[1]
        
#resolução do sistema P(Transposto) . x = x
P_acum= LA.matrix_power(P, estados*100)
P_acum = np.add.accumulate(P_acum[0])
P_acum = P_acum.tolist()


#Calculando a média de bolsas de sangue no estoque K

fp = [0] * estados
for i in range(len(P_acum) -1):
    fp[i] += P_acum[i+1] - P_acum[i]

fp_n = [0] * estados                     #normalização de fp prevendo erros numéricos
for i in range(estados):
    fp_n[i] += fp[i]/sum(fp)
    

Ex = [0] * estados
for i in range(estados):
    Ex[i] += list(range(estados))[i]*fp_n[i]

V_Ex = sum(Ex)                          #nº esperado de bolsas de sangue é o valor esperado da função de prob.



#-------------------------------------------------------------------------------------------------------
#SAÍDA 
#consertar aqui :/

#print("ações: ", mem_acoes)
print('')
print("Optimal policy: Send 2 bus from state 0 to state ",mem_acoes.count('2bus')-1-n0)
print("send 1 bus from state",mem_acoes.count('2bus')-n0, "to state", mem_acoes.count('2bus')+mem_acoes.count('1bus')-1-n0)
#print("Valor esperado do estoque E(n): ", V_Ex)
#print("iterações: ", it)
#print("tempo de execução: ", round(t_final - t_inicial,1), " segundos")



#-----------------------------------------------------------------
#GRAFICOS

#grafico da probabilidade acumulada dos estados ao longo dos estados
#plt.plot(range(1-n0,estados+1-n0),P_acum)
#plt.title("Probabilidade acumulada ao longo dos estados")
#plt.ylabel("Prob.")
#plt.xlabel("Estados")
#plt.grid(True)

#grafico do logaritmo do inverso da probabilidade acumulada dos estados 
#ln_P_acum = []
#for Pn in P_acum:
#    ln_P_acum.append(math.log(Pn**-1))

#plt.plot(range(1-n0,estados+1-n0),ln_P_acum)
#plt.title("Probabilidade acumulada ao longo dos estados")
#plt.ylabel("Prob.")
#plt.xlabel("Estados")
#plt.grid(True)

#grafico da evolução do custo C ao longo dos estados
#plt.plot(range(1-n0,len(C)+1-n0),C)
#plt.title("Evolução do custo ao longo dos estados")
#plt.ylabel("Custo")
#plt.xlabel("Estados")
#plt.grid(True)

#grafico da evolução do custo Cf ao longo dos estados
#plt.plot(range(1-n0,len(Cf)+1-n0),Cf)
#plt.title("Evolução do custo ao longo dos estados")
#plt.ylabel("Custo de falta")
#plt.xlabel("Estados")
#plt.grid(True)

#grafico da evolução do custo Ce ao longo dos estados
#plt.plot(range(1-n0,len(Ce)+1-n0),Ce)
#plt.title("Evolução do custo ao longo dos estados")
#plt.ylabel("Custo de estoque")
#plt.xlabel("Estados")
#plt.grid(True)


#grafico da taxa de chegada lambda ao longo dos estados
v_lambda = [lamb]*estados
for i in range(mem_acoes.count('2bus')-1-n0):
    v_lambda[i] += c
for i in range(mem_acoes.count('2bus')- n0 +mem_acoes.count('1bus')-1):
    v_lambda[i] += c

plt.plot(range(1-n0,len(v_lambda)+1-n0),v_lambda)
plt.title("Policy")
plt.ylabel("Lambda")
plt.xlabel("States")
plt.grid(True)
plt.show()

