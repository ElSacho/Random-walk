import pygame
import numpy as np
import cv2
import math 
import scipy

# ===============================================
# === PARTIE POUR LA LABELISATION D'UN IMAGE ===
# ===============================================

def dessinerLabel(labels):
    
    global continuer 
    global ecran
    global labelActuel
    global nbrLabels
    
    # on recupere les actions de l'utilisateur de l'ordinateur
    ev = pygame.event.get()
    # proceed events
    for event in ev:
        # s'il a cliqué sur un point on le labelise dans le label actuel
        if event.type == pygame.MOUSEBUTTONDOWN:
            action = np.array(pygame.mouse.get_pos())
            remplir_vecteur_label(labels, action, labelActuel)
            pygame.draw.circle(ecran, (255, 255, 255), (action[0], action[1]), 5)
            print(action)
            return
        
        # s'il ferme l'interface on quitte tout
        if event.type == pygame.QUIT:
            continuer = False
            break
        
        # s'il  appuie sur une touche, on passe à la labelisation suivante
        if event.type == pygame.KEYDOWN:
            labelActuel +=1
            if labelActuel>nbrLabels:
                break
            print(f"Labelisation de la section numero {labelActuel}")

def remplir_vecteur_label(labels, action, serie, size = 1):
    
    global ecran
    
    # On remplie le vecteur des labels en fonction des endroits ou l'on a cliqué
    # On peut modifier la valeur de "size" pour qu'on clique labelise une zone autour du point de clique
    labels[action[1],action[0]]=serie
    for i in range(size):
        for j in range(size):
            if i+action[1]<labels.shape[0] and j+action[0]<labels.shape[1]:
                labels[i+action[1],j+action[0]]=serie
  
def creer_les_labels(img, nbr_labels):

    global continuer
    global ecran
    global labelActuel
    global nbrLabels
    
    nbrLabels = nbr_labels
    labelActuel = 1

    # On initialise l'interface de labelisation avec les dimensions de l'image
    pygame.init()
    
    ecran = pygame.display.set_mode((300,300))

    image = pygame.image.load(img).convert_alpha()

    h = image.get_height()
    w = image.get_width()

    labels = np.zeros((h,w))
    ecran = pygame.display.set_mode((w,h))
    print(f"Labelisation de la section numero {labelActuel}")

    # Tant qu'on veut continuer et qu'on a pas tout labelisé on labelise
    continuer = True
    while continuer and labelActuel<=nbrLabels:
        ecran.blit(image, (0, 0))
        dessinerLabel(labels)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT :
                continuer = False
        pygame.display.flip()

    pygame.quit()
    return labels

# ===============================================
# === PARTIE POUR L'ALGORITHME RANDOM WALK ===
# ===============================================


def matrix_to_vector(matrice):
    return matrice.flatten()

def fonction_weight(gi,gj, beta):
    return math.exp(-beta*math.dist(gi,gj)*math.dist(gi,gj))

def get_matrice_L(matrice,beta):
    
    #Initialisation de la matrice finale avec des zeros partout
    nx, ny , nz = matrice.shape
    L = np.zeros((nx*ny,nx*ny))
    
    # on remplit les points qui doivent l'être
    for i in range(nx):
        for j in range(ny):
            
            valeurCentrale=0
            
            # On prend tous les voisins, potentielle optimisation ici car on calcul plusieurs fois les mêmes valeurs, 
            # les poids étant symetriques
            
            for nex,ney in neighboor_all(i,j,ny,nx):
                
                weight = - fonction_weight(matrice[i,j],matrice[nex,ney], beta)
                valeurCentrale+=weight
                id1 = i*ny+j
                id2 = nex*ny+ney
                
                #Comme la matrice est symétrique on peut ajoute les valeurs aux deux endroits directement
                
                L[id1,id2] = weight
                L[id2,id1] = weight
                
            # On ajoute la somme des poids avec les voisins pour le point central
            L[i*ny+j,i*ny+j] = - valeurCentrale
            
    return L   

def neighboor_all(i,j, ny, nx):
    
    neighboor =[]
    list_idx = [ i - k + 1 for k in range(3) if (i - k + 1 >= 0 and i - k + 1 <= nx - 1 ) ]
    list_idy = [ j - k + 1 for k in range(3) if (j - k + 1 >= 0 and j - k + 1 <= ny - 1 ) ]
    
    for idx in list_idx:
        for idy in list_idy:
            neighboor.append([idx, idy])
            
    return neighboor    
  
def get_permutation(vector_label):
    
    global nbr_points_labelises
    # On commence par compter le nombre de vector
    nbr_points_labelises = 0
    for i in range(vector_label.shape[0]):
            if vector_label[i] !=0:
                nbr_points_labelises+=1

    idx_label = 0
    pointeurActuel = 0
    tabPermutation = []
    
    # tant qu'on a pas mis tous les labels devant
    while idx_label < nbr_points_labelises:
        
        # Si on est pas labelise va voir le points suivants
        if vector_label[ pointeurActuel ] == 0:
            pointeurActuel += 1
            
        else :
            #On echange les les valeurs entre le premier point non labelise et le point labelise
            temp = vector_label[ pointeurActuel ]
            vector_label[ pointeurActuel ] = vector_label[ idx_label ]
            vector_label[ idx_label ] = temp
            
            #On ajoute la permutation dans le tableau et augmente le nombre de labels presents dans la nouvelle liste
            tabPermutation.append([pointeurActuel,idx_label])
            idx_label += 1
            
    return tabPermutation, vector_label

def getMatricePermutation(IDE,perm):
    
    #pour tous les echanges a faire on effectue l'echange
    for echange in perm:
        depart = echange[0]
        arrivee = echange[1]
        temp = IDE[:,depart].copy()
        IDE[:,depart] = IDE[:,arrivee]
        IDE[:,arrivee] = temp
        
    return IDE

def permuteL(L,perm):
    
    #On trouve la premiere matrice de permutation
    id =  np. eye(L.shape[0])
    P1 = getMatricePermutation(id,perm)

    # On trouve la seconde matrice de permutation
    id2 = np. eye(L.shape[0]) 
    P2 = getMatricePermutation(id2,perm[::-1])

    #On effectue le calcul
    #C'est la phase la plus longue de l'algorithme
    return P1@L@P2
        
def getMatricesToSolve(L,vectorLabelOrdone):
    
    global nbr_points_labelises
    global nbrLabels

    # On extrait Lu et B de L
    Lu = L[nbr_points_labelises:,nbr_points_labelises:]
    B = L[:nbr_points_labelises:, nbr_points_labelises:]
    
    M = np.zeros((nbr_points_labelises,nbrLabels))
    
    i = 0
    while i<len(vectorLabelOrdone) and vectorLabelOrdone[i]!=0:
        k = int(vectorLabelOrdone[i])
        M[i,k-1]=1
        i+=1
    
    return Lu, B, M

def solve(Lu, B, xM):
    print(np.linalg.det(Lu))
    xU = - np.linalg.inv(Lu) @ B.T @xM
    return xU

def solve2(Lu,B,M):
    K = M.shape[1]
    xU=[]
    for idx in range(K):
            pot = scipy.sparse.linalg.spsolve(
                Lu, -B.T @ M[:,idx])
            xU.append(pot)
    return xU

def permutationVecteur(x, perm):

    #Pour toutes les permutations on echange de place les deux donnees
    for echange in perm:
        depart = echange[0]
        arrivee = echange[1]
        
        temp = x[depart].copy()
        x[depart] = x[arrivee]
        x[arrivee] = temp
        
    return x

def transformEnLabel(M,xu):
    x = []
    
    # On parcourt M pour applatir obtenir un vecteur avec le numéro du label a tous les endroits
    for el in M:
        idx = 0
        for val in el:
            if val==1:
                x.append(idx+1)
                break
            idx+=1
            
    # On parcourt xu pour trouver la segmentation la plus probable selon l'algorithme
    for el in xu:
        idx = 0
        m = el[0]
        for k in range(1,len(el)):
            if el[k]>m:
                idx = k
        x.append(idx+1)
        
    return np.array(x)

def resize_vector_to_matrix(x, img):
    nx, ny, nz = img.shape
    return x.reshape((nx,ny))

def random_walk(img, matrice_label, beta):
    
    # On construit la matrice L
    L = get_matrice_L(img/255,beta)
    print("Matrice L générée")
    
    # On permute cette matrice pour placer les vecteurs labelises devant
    vector_labelise = matrix_to_vector(matrice_label) 
    perm, vectorLabelise_ordonne = get_permutation(vector_labelise)
    L = permuteL(L,perm)
    print("Permutation effectuée")

    # On extrait les matrices necessaires a la resolution de l'equation
    Lu , B , M = getMatricesToSolve(L,vectorLabelise_ordonne)
    print("Extraction réalisée")

    # On resout le systeme pour obtenir la matrice des probabilites d'appartenir a chaque label
    # Essayer de modifier cette fonction solve pour aller plus vite surtout pour des dimensions plus grandes
    xu = solve(Lu, B, M)
    print("Resolution du systeme ok")
    
    # On reorganise les resultats en transformant les probabilites en labelisation 
    # et reorganise les resultats dans l'ordre
    x = transformEnLabel(M,xu)
    print("Analyse des resultats effectuée")
    x = permutationVecteur(x, perm[::-1])
    print("Resultats reordonnés")
    imgLabel = resize_vector_to_matrix(x, img)
    print("Labelisation des données terminées")
    
    # On renvoit un array de la taille de l'image à labeliser ou toutes les valeurs sont le label 
    # auquel les points appartiennent probablement
    return imgLabel

# ===============================================
# === PARTIE POUR L'AFFICHAGE DES RESULTATS ===
# ===============================================


def drawResult(imgLabel,image):
    
    global nbrLabels

    # on recupere l'image initiale pour recuperer un vecteur de la taille souhaitée
    convert = cv2.imread(image)
    
    # pour tous les pixels de l'image, on remplace par une valeur proportionnelle au label 
    shape = imgLabel.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            val = int((int(imgLabel[i][j])*255/int(nbrLabels)))
            convert[i][j]=[val,val,val]

    # on affiche l'image finale
    cv2.imshow('labels',convert)

    cv2.waitKey(0)

    cv2.destroyAllWindows() 
    
def main(image, beta):

    #On labelise a la main les donnees
    print("Combien de labels y a-t-il ? : ", end='')
    nbrLabels = int(input())
    print("Vous pouvez labeliser l'image. \n Pour chaque label cliquez sur les points présents dans le label puis appuyez sur 'espace' pour passer au label suivant")
    labels = creer_les_labels(image, nbrLabels)
    print("Labelisation ok")
    
    # On effectue le random walk sur l'image
    img = cv2.imread(image)
    imgLabel = random_walk(img, labels,beta)
    
    # On affiche le resultat    
    drawResult(imgLabel,image)
    
main("imagesTest/test04.jpg",10)