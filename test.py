import pygame
import numpy as np
import cv2

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
    
    for i in range(size):
        for j in range(size):
            if i+action[1]<labels.shape[0] and j+action[0]<labels.shape[1]:
                labels[i+action[1],j+action[0]]=serie

    
def labelisation(img):

    global continuer
    global ecran
    global labelActuel
    global nbrLabels
    
    nbrLabels = 4 # un input en vrai
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

labels = labelisation("imTest.png")
print(labels)
img = cv2.imread("imTest.png")
print(img.shape)

#[0] -> les lignes
#[1] -> les colonnes

for i in range(img.shape[1]):
    for j in range(img.shape[0]):
        if labels[j,i]!=0:
            img[i,j]=[255,255,255]
            
cv2.imshow('labels',img)

cv2.waitKey(0)

cv2.destroyAllWindows()

