import pygame
import numpy as np
import cv2
import math 

def dessinerLabel(labels, int=1):
    
    global continuer 
    global ecran
    global labelActuel
    global nbrLabels
    
    ev = pygame.event.get()
    # proceed events
    for event in ev:
        if event.type == pygame.MOUSEBUTTONDOWN:
            action = np.array(pygame.mouse.get_pos())
            labeliser(labels, action, labelActuel)
            pygame.draw.circle(ecran, (255, 255, 255), (action[0], action[1]), 5)
            print(action)
            return
        if event.type == pygame.QUIT:
            continuer = False
            break
        if event.type == pygame.KEYDOWN:
            labelActuel +=1
            if labelActuel>nbrLabels:
                break
            print(f"Labelisation de la section numero {labelActuel}")

      
def labeliser(labels, action, serie, size =5):
    
    global ecran
    labels[action[1],action[0]]=serie
    return
    for i in range(size):
        for j in range(size):
            if i+action[1]<labels.shape[0] and j+action[0]<labels.shape[1]:
                labels[i+action[1],j+action[0]]=serie

    
def labelisation(img):

    global continuer
    global ecran
    global labelActuel
    global nbrLabels
    
    nbrLabels = int(input()) # un input en vrai
    labelActuel = 1

    pygame.init()
    
    ecran = pygame.display.set_mode((300,300))

    image = pygame.image.load(img).convert_alpha()

    h = image.get_height()
    w = image.get_width()
    shape = (w,h)
    labels = np.zeros((h,w))
    ecran = pygame.display.set_mode((w,h))
    print(f"Labelisation de la section numero {labelActuel}")

    continuer = True
    c=0
    while continuer and labelActuel<=nbrLabels:
        ecran.blit(image, (0, 0))
        dessinerLabel(labels, labelActuel)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT :
                continuer = False
        pygame.display.flip()

    pygame.quit()
    return labels



def matrix_to_vector(matrice):
    nx, ny = matrice.shape
    vector = []
    for i in range(nx):
        for j in range(ny):
            vector.append(matrice[i,j])
    return np.array(vector)

def fonction_weight(gi,gj, beta):
    #return abs(gi-gj)
    return math.exp(-beta*math.dist(gi,gj)*math.dist(gi,gj))
    return math.exp(-beta*(gi-gj)*(gi-gj))



def get_matrice_L(matrice,beta):
    nx, ny , nz = matrice.shape
    L = np.zeros((nx*ny,nx*ny))
    for i in range(nx):
        for j in range(ny):
            s=0
            for nex,ney in neighboor_all(i,j,ny,nx):
                weight = - fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                s+=weight
                id1 = i*ny+j
                id2 = nex*ny+ney
                L[id1,id2]=weight
                L[id2,id1]=weight
                s+=weight
            L[i*ny+j,i*ny+j]=-s/2
    #for diag in range(nx*ny):
    #    L[diag,diag]=sum(L[diag,:])
    return L   

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


def get_matrice_L2(matrice,beta):
    nx, ny , nz = matrice.shape
    L = np.zeros((nx*ny,nx*ny))
    for i in range(nx):
        s=0
        for j in range(ny):
            s=0
            for nex,ney in neighboor_before(i,j,ny):
                weight = fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                s+=weight
                id1 = i*ny+j
                id2 = nex*ny+ney
                L[id1,id2]=weight
                L[id2,id1]=weight
    for diag in range(nx*ny):
        if diag%ny == 0:
            print(diag)
        L[diag,diag]=sum(L[diag,:])
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

#Ne fonctionne pas comme je veux 
#En fait il faut trouver la matrice de permutation et permuter les valeurs

def permute(L,perm):
    c=0
    k = len(perm)
    for echange in perm:
        depart = echange[0]
        arrivee = echange[1]
        c+=1
        #print(c/k)
        #on change dans L
        temp = L[:,depart].copy()
        L[:,depart] = L[:,arrivee]
        L[:,arrivee] = temp
    return L 

def permuteL(L,perm):
    id =  np. eye(L.shape[0])
    P1 = permute(id,perm)
    print('P1')
    id2 = np. eye(L.shape[0]) 
    P2 = permuteInverse(id2,perm)
    print('Plus que le calcul')
    L = P1@L@P2
    return L,P1,P2

def permuteInverse(L,perm):
    print("aaa")
    print(perm[::-1])

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
    nombreLot = int(vectorLabelOrdone[nombreLabel])
    while vectorLabelOrdone[nombreLabel]!=0:
        if vectorLabelOrdone[nombreLabel]>nombreLot:
            nombreLot=vectorLabelOrdone[nombreLabel]
        nombreLabel+=1
    Lu = L[nombreLabel:,nombreLabel:]
    B = L[:nombreLabel:, nombreLabel:]
    print(nombreLabel)
    print(nombreLot)
    
    xM = np.zeros((nombreLabel,int(nombreLot)))
    
    i = 0
    while i<len(vectorLabelOrdone) and vectorLabelOrdone[i]!=0:
        k = int(vectorLabelOrdone[i])
        xM[i,k-1]=1
        i+=1
    
    return Lu, B, xM

def solve(Lu, B, xM):
    print(Lu)
    print(np.linalg.det(Lu))
    xU = - np.linalg.inv(Lu) @ B.T @xM
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
    L = get_matrice_L(img/255,beta)
    print("Step 1 ok")
    perm, vectorLabelOrdone = get_permutation(vectorLab)
    print("Step 2 ok")
    print(perm)
    L,P1,P2 = permuteL(L,perm)
    print("Step 3 ok")
    Lu , B , xM = getMatricesToSolve(L,vectorLabelOrdone)
    #xm est ok 
    print("Step 4 ok")
    xU = solve(Lu, B, xM)
    print(f'xu : \n {xU}')
    print("Step 5 ok")
    x = transformEnLabel(xM,xU)
    print(f'x : \n {x}')
    print("Step 6 ok")
    x = permInverse(x, perm)
    print("Step 7 ok")
    imgLabel = fromLineToDoubleLine(x, img)
    print("Finish")
    return imgLabel

def drawResult(imgLabel,image):
    global nbrLabels
    
    tab = []

    for lign in imgLabel:
        temp = []
        for el in lign:
            temp.append(el)
        tab.append(temp.copy())

    convert = cv2.imread(image)
    print(convert.shape)
    labelisee= np.array(imgLabel)
    shape = labelisee.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = int((int(labelisee[i][j])*255/int(nbrLabels)))
            convert[i][j]=[val,val,val]


    cv2.imshow('labels',convert)

    cv2.waitKey(0)

    cv2.destroyAllWindows() 
    
    return

    lab = tab
    convert = cv2.imread(image)
    print(convert.shape)
    print(np.array(imgLabel).shape)
    print(len(lab[0]))
    print(len(lab[1]))
    for i in range(len(lab[0])):
        for j in range(len(lab[1])):
            val = int((int(lab[i][j])*255/int(nbrLabels)))
            convert[i][j]=[val,val,val]

    cv2.imshow('labels',convert)

    cv2.waitKey(0)

    cv2.destroyAllWindows() 

    
def labelisationMain(image, beta):

    labels = labelisation(image)
    print("Leblisation ok")
    img = cv2.imread(image)
    print("Lecture image ok")

    imgLabel = main(img, labels,beta)
    print("correction ok")
    
    drawResult(imgLabel,image)
    
labelisationMain("test04.jpg",10)