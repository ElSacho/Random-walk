import numpy as np
import math
import scipy

def matrix_to_vector(matrice):
    nx, ny = matrice.shape
    vector = []
    for i in range(nx):
        for j in range(ny):
            vector.append(matrice[i,j])
    return np.array(vector)

def fonction_weight(gi,gj, beta):
    #return abs(gi-gj)
    #return math.exp(-beta*math.dist(gi,gj)*math.dist(gi,gj))
    return math.exp(-beta*(gi-gj)*(gi-gj))

def get_matrice_poids(matrice, beta):
    nx, ny = matrice.shape
    we = []
    for i in range(nx):
        for j in range(ny):
            w = []
            for nex,ney in neighboor_before(i,j,ny):
                weight = fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                w.append((weight,[i,j],[nex,ney]))
            we.append(w.copy())
    return we

def get_matrice_L2(matrice,beta):
    nx, ny = matrice.shape
    L = np.zeros((nx*ny,nx*ny))
    for i in range(nx):
        s=0
        for j in range(ny):
            s=0
            for nex,ney in neighboor_before(i,j,ny):
                weight = - fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                s+=weight
                id1 = i*ny+j
                id2 = nex*ny+ney
                L[id1,id2]=weight
                L[id2,id1]=weight
    for diag in range(nx*ny):
        L[diag,diag]= - sum(L[diag,:])
    return L 

def get_matrice_L(matrice,beta):
    nx, ny = matrice.shape
    L = np.zeros((nx*ny,nx*ny))
    for i in range(nx):
        s=0
        for j in range(ny):
            s=0
            for nex,ney in neighboor_all(i,j,ny,ny):
                weight = - fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                s+=weight
                id1 = i*ny+j
                id2 = nex*ny+ney
                L[id1,id2]=weight
                L[id2,id1]=weight
                s+=weight
            L[i*ny+j,i*ny+j]=- s/2
    #for diag in range(nx*ny):
    #    L[diag,diag]=sum(L[diag,:])
    return L     
    
def neighboor_before(i,j, ny):
    neigh=[]
    if i-1>=0:
        if j-1>=0:
            neigh.append([i-1, j-1])
        neigh.append([i-1, j])
        if j+1<ny:
            neigh.append([i-1,j+1])
    if j-1>=0:
        neigh.append([i,j-1])
    return neigh

def neighboor_all(i,j, ny, nx):
    neigh=[]
    if i-1>=0:
        if j-1>=0:
            neigh.append([i-1, j-1])
        neigh.append([i-1, j])
        if j+1<ny:
            neigh.append([i-1,j+1])
    if j-1>=0:
        neigh.append([i,j-1])
    if j+1<ny:
        neigh.append([i,j+1])
    if i+1<nx:
        if j-1>=0:
            neigh.append([i+1, j-1])
        neigh.append([i+1, j])
        if j+1<ny:
            neigh.append([i+1, j+1])
    return neigh
    
def get_permutation2(vectorLabRowAndColumns):
    #On parcours la liste, on on est labelisé on passe au suivant, sinon, on echange avec un element en commencant par la fin et on remonte la liste jusqua ce qu on ai le bon nombre d elements
    nombreDeLabel = 0
    vectorLabel = []
    for i in range(vectorLabRowAndColumns.shape[0]):
        for j in range(vectorLabRowAndColumns.shape[1]):
            lab = vectorLabRowAndColumns[i,j]
            if lab !=0:
                nombreDeLabel+=1
            vectorLabel.append(lab)
    n = len(vectorLabel)
    vectorLabel = np.array(vectorLabel)
    nombreDeRejets=0
    indice=0
    tabPermutation=[]
    while indice<nombreDeLabel: #tant qu on a pas labelise tous les points
        if vectorLabel[indice]!=0: #si on est labelisee c est bon
            indice+=1
        else : # sinon on le met en fin de liste
            temp = vectorLabel[indice]
            vectorLabel[indice] = vectorLabel[n-1-nombreDeRejets]
            vectorLabel[n-1-nombreDeRejets] = temp
            tabPermutation.append([indice,n-1-nombreDeRejets])
            nombreDeRejets+=1
    return tabPermutation, vectorLabel

    
def get_permutation(vectorLabRowAndColumns):
    #On parcours la liste, on on est labelisé on passe au suivant, sinon, on echange avec un element en commencant par la fin et on remonte la liste jusqua ce qu on ai le bon nombre d elements
    nombreDeLabel = 0
    vectorLabel = []
    for i in range(vectorLabRowAndColumns.shape[0]):
        for j in range(vectorLabRowAndColumns.shape[1]):
            lab = vectorLabRowAndColumns[i,j]
            if lab !=0:
                nombreDeLabel+=1
            vectorLabel.append(lab)
    n = len(vectorLabel)
    nombreLabelise=0
    pointeurActuel = 0
    tabPermutation=[]
    while nombreLabelise<nombreDeLabel:
        if vectorLabel[pointeurActuel]==0:
            pointeurActuel+=1
        else :
            temp = vectorLabel[pointeurActuel]
            vectorLabel[pointeurActuel]=vectorLabel[nombreLabelise]
            vectorLabel[nombreLabelise]=temp
            tabPermutation.append([pointeurActuel,nombreLabelise])
            nombreLabelise+=1
    return tabPermutation, vectorLabel

            


#Ne fonctionne pas comme je veux 
#En fait il faut trouver la matrice de permutation et permuter les valeurs

def permute(L,perm):
    for echange in perm:
        depart = echange[0]
        arrivee = echange[1]
        
        #on change dans L
        temp = L[:,depart].copy()
        L[:,depart] = L[:,arrivee]
        L[:,arrivee] = temp
    return L 


def permuteL(L,perm):
    id =  np. eye(L.shape[0]) 
    P1 = permute(id,perm)
    id2 = np. eye(L.shape[0]) 
    P2 = permuteInverse(id2,perm)
    L = P1@L@P2
    return L,P1,P2

def permuteInverse(L,perm):
    for echange in perm[::-1]:
        depart = echange[0]
        arrivee = echange[1]
        
        #on change dans L
        temp = L[:,depart].copy()
        L[:,depart] = L[:,arrivee]
        L[:,arrivee] = temp
    return L 
        
def getMatricesToSolve(L,vectorLabelOrdone):
    nombreLabel = 0
    nombreLot = vectorLabelOrdone[nombreLabel]
    while vectorLabelOrdone[nombreLabel]!=0:
        if vectorLabelOrdone[nombreLabel]>nombreLot:
            nombreLot=vectorLabelOrdone[nombreLabel]
        nombreLabel+=1
    Lu = L[nombreLabel:,nombreLabel:]
    B = L[:nombreLabel:, nombreLabel:]
    
    xM = np.zeros((nombreLabel,nombreLot))
    
    i = 0
    while i<len(vectorLabelOrdone) and vectorLabelOrdone[i]!=0:
        k = vectorLabelOrdone[i]
        xM[i,k-1]=1
        i+=1
    
    return Lu, B, xM

def solve(Lu, B, xM):
    print(Lu)
    print(np.linalg.det(Lu))
    xU = - np.linalg.inv(Lu) @ B.T @xM
    return xU

def solve2(Lu,B,xM):
    K = xM.shape[1]
    print(K)
    xU=[]
    print(xM)
    for idx in range(K):
            print('tes dans la matrice')
            print(xM[:,idx])
            pot = scipy.sparse.linalg.spsolve(
                Lu, -B.T @ xM[:,idx])
            xU.append(pot)
    return xU

def permInverse(x, perm):
    print(x)
    print(perm[::-1])
    for echange in perm[::-1]:
        depart = echange[0]
        arrivee = echange[1]
        
        temp = x[depart].copy()
        x[depart] = x[arrivee]
        x[arrivee] = temp
    return x

def transformEnLabel(xM,xU):
    x = []
    for el in xM:
        idx = 0
        for val in el:
            if val==1:
                x.append(idx+1)
                break
            idx+=1
    for el in xU:
        idx = 0
        m = el[0]
        for k in range(1,len(el)):
            if el[k]>m:
                idx = k
        x.append(idx+1)
    return np.array(x)

def fromLineToDoubleLine(x, img):
    shape = img.shape
    imgLabel = []
    for i in range(shape[0]):
        tab = []
        for j in range(shape[1]):
            tab.append(x[i*shape[1]+j])
        imgLabel.append(tab.copy())
    return np.array(imgLabel)

def main(img, vectorLab, beta):
    L = get_matrice_L2(img,beta)
    perm, vectorLabelOrdone = get_permutation2(vectorLab)
    print(f"L:\n{L}")
    L,P1,P2 = permuteL(L,perm)
    print(f"perm:\n{perm}")
    print(f"L permute:\n{L}")
    Lu , B , xM = getMatricesToSolve(L,vectorLabelOrdone)
    print(f"Lu :\n{Lu}")
    print(f"B :\n{B}")
    print(f"xM :\n{xM}")
    xU = solve(Lu, B, xM)
    print(f"xU :\n{xU}")
    t= [sum(xU[i]) for i in range(len(xU))]
    print(f't : {t}')
    print()
    print(xM)
    print("ca")
    #print(Lu@xU)
    #print(-B.T@xM)
    #print(xU)
    x = transformEnLabel(xM,xU)
    print("asadx")
    print(x)
    x = permInverse(x, perm)
    imgLabel = fromLineToDoubleLine(x, img)
    print(imgLabel)
    return imgLabel
    

mat = np.array([
    [1,1,20]
    ,[1,1,20]
    ,[1,1,20]
    ,[1,1,1]
])

vectorLab = np.array([
    [0,1,0]
    ,[1,1,2]
    ,[0,1,2]
    ,[0,0,1]
])

mat2 = np.array([
    [2,1,20]
    ,[12,20,12]
    ,[12,20,12]
])

vectorLab2 = np.array([
    [1,1,1]
    ,[2,0,0]
    ,[0,0,0]
])

mat3 = np.array([
    [2,1]
    ,[4,3]
])

vectorLab3 = np.array([
    [0,0]
    ,[1,0]
])


main(mat, vectorLab, 1)


"""
LU = np.array([
    [0,0,1],
    [1,10,0],
    [11,29,0]
])


B = np.array([
    [1,0,2],
    [32,11,1],
    [19,8,0]
])

M = np.array([
    [1,0],
    [1,0],
    [0,1]
])


xu = - np.linalg.inv(LU) @ B.T @M
LUinv = np.linalg.inv(LU)
BT = B.T 

print(f"xu : \n: {xu}")
print(f"LUinv : \n: {LUinv}")
print(f"BT : \n: {BT}")
print(f"CalculInv : \n: {LUinv@LU}")
"""