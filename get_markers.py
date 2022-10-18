import pygame
import numpy as np
import cv2
import math 
import scipy

# ===============================================
# === PARTIE POUR LA LABELISATION D'UN IMAGE ===
# ===============================================

def dessinerLabel(labels, size=1):
    
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
            remplir_vecteur_label(labels, action, labelActuel, size)
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

    
def creer_les_labels(img, nbr_labels, size=1):

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
        dessinerLabel(labels, size)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT :
                continuer = False
        pygame.display.flip()

    pygame.quit()
    return labels

def writeIn(labels, doc):
    fichier = open(doc, "w")
    fichier.write("labels=")
    lab = []
    for i in range(labels.shape[0]):
        temp = []
        for j in range(labels.shape[1]):
            temp.append(int(labels[i,j]))
        lab.append(temp.copy())
    fichier.write(str(lab))
    fichier.close()
    
def writeRGBIn(img, doc):
    fichier = open(doc, "w")
    fichier.write("labels=")
    print(img.shape)
    lab = []
    for i in range(img.shape[0]):
        temp = []
        for j in range(img.shape[1]):
            RGB=0
            for k in range(img.shape[2]):
                RGB+=int(img[i,j,k])
            RGB = RGB/img.shape[2]
            temp.append(RGB)
        lab.append(temp.copy())
    fichier.write(str(lab))
    fichier.close()

def create_label(image,nbrLabels, size=1):
    #On labelise a la main les donnees
    #print("Combien de labels y a-t-il ? : ", end='')
    #nbrLabels = int(input())
    print("Vous pouvez labeliser l'image. \n Pour chaque label cliquez sur les points présents dans le label puis appuyez sur 'espace' pour passer au label suivant")
    labels = creer_les_labels(image, nbrLabels, size)
   # writeIn(labels, 'label.py')
    img = cv2.imread(image)
   # print(img)
   # writeRGBIn(img, 'image.py')
    new_label = np.zeros((labels.shape[0],labels.shape[1],3))
    for i in range(3):
        new_label[:,:,i]=labels
    return new_label, img
    

#create_label("imagesTest/test04.jpg")