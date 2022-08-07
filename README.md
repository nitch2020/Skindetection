# Skindetection
Determiner les caracteristiques des pixels peau et non peau de l'image dans l'espace de couleur LAB
# RESUME
Le système de détection de la peau se compose de deux phases : phase d'apprentissage, détection de la peau Tout d'abord, la phase d'apprentissage implique différentes étapes telles que la sélection d'ensembles de données d'images de peau, la sélection de l'espace
colorimétrique approprié et l'identification des paramètres pour le classificateur de peau. La deuxième phase est la détection de la peau. Il identifie le pixel de peau à partir de caractéristiques de peau formées données. Dans cette phase, l'image est transformée en un espace colorimétrique différent LAB. Ensuite, la caractéristique est extraite pour représenter la région de la peau. Les fonctionnalités sont utilisées pour classer les pixels en pixels de peau ou non. Et enfin la représentation des histogrammes peau et non peau.La détection de la peau est basée sur l'histogramme. En réalité on aura 2 histogrammes: un histogramme qui présente les pixels peau et un autre qui présente les pixels non peau.
liens base images d'entraînement , de test et leurs masques: https://drive.google.com/drive/folders/1L5-z4q1nmLfkJkX4UyCFjgQR_hzy_lDO?usp=sh
aring
# La détection de la couleur de la peau humaine est importante dans de nombreuses applications. Il existe diverses applications basées sur la peau dans plusieurs domaines, à savoir l'analyse des gestes, la reconnaissance faciale, le suivi des personnes et la détection de la nudité, la récupération d'images basée sur le contenu. En effet, dans le cadre du cours de vision par ordinateur, cette tâche nous à été assignée. Et pour parvenirà des résultats attendus, nous avons utilisé plusieurs techniques de vision par ordinateur en exploitant la librairie Opencv de Python. Nous allons donc vous présenter
les différentes étapes de ce travail dans la suite de ce document.
# OUTILS UTILISÉS
➢ Opencv: Open Source Computer Vision. C’est une bibliothèque graphique libre spécialisée dans le traitement d’images. Elle nous a permis entre autre de lire nos images, de les afficher, de passer d’un espace de couleur à un autre et bien d’autres taches.
➢ Numpy: c’est la bibliothèque la plus populaire de calcul scientifique en python. Elle nous a principalement servi pour la gestion de nos tableaux d’images.
➢ Matplotlib: c’est une librairie compréhensive pour créer des visualisations statiques, animées et interactives en python.
➢ Colab: c’est un produit de google search qui permet d’écrire et d'exécuter le code python de son choix par le biais du navigateur. C’est un environnement
particulièrement adapté au machine learning et l’analyse de données.
# IMPLEMENTATION DU MODELE
Pour réaliser ce projet, nous avons utilisé une base d'images contenant des exemples de pixels peau et de
pixels non-peau accessible à travers ce lien suivant: https://drive.google.com/uc?id=1MnBW_OJqrTmzwc23YI5NK_y_l4zk9JGJ
Cette base d'images contient des images d'apprentissage pour entraîner notre modèle (train) et une base d'images de test (test) pour valider les résultats. Chaque exemple a une image d'origine et un masque peau correspondant. En outre, nous avons collecté d’autres images à travers le net pour renforcer
l'entraînement de notre modèle et ces images sont pour la plupart des images non peau, permetant ainsi d'augmenter la probabilité des pixels non peau.
lien:https://drive.google.com/drive/folders/1S31oJ8es5MkR9EAlryVXDOaZvlPsBLEo?usp=sharing
# Etape 1: Récupération des images et masques dans des tableaux
Après avoir établi notre base d’image, nous créons un tableau dans lequel on ajoute chacune des images originales d'entraînement et un autre tableau dans lequel on ajoute le masque de chaque image. Grâce à la méthode shape pour les tableaux à n dimensionsde numpy, nous pouvons visualiser les caractéristiques des tableaux obtenus.
# Étape 2: Afficher quelques images et leurs masques
Pour celà nous récupérons une série d'images de façon aléatoire à chaque exécution. Ensuite nous les affichons grâce à certaines méthodes de matplotlib tel que “suplot” et la méthode “imshow” de opencv.
# Etape 3: Changement de l’espace colorimétrique (RGB to LAB)
# RVB fonctionne sur trois canaux : rouge, vert et bleu. LAB est une conversion des mêmes informations en un composant de luminosité L* et deux composants de couleur - a* et b*.Danscette étape on change l’espace de couleur de l’image en passant de l’espace RGB à l’espace Lab grace à la fonction de opencv cv.COLOR_RGB2LAB.
Étape 4: Récupération et réduction dans l'espace Lab avec les intervalles de a et b convertis 
Dans cette partie, après avoir converti nos images dans l’espace LAB, on utilise seulement les canaux a et b sans tenir compte du canal L pour la luminance. Ensuite on procède à la réduction de la quantification, c'est-à-dire de réduire l’intervalle des valeurs de pixels dans chaque canal.
# Étape 7: Calcul des histogrammes peau et non peau
Cette partie consiste à utiliser les images qu’on a converties et réduire dans l’espace Lab et leur masque pour calculer les histogrammes peau et non peau. Les histogrammes permettent de définir les valeurs de pixels correspondant à la peau et les valeurs de pixels non peau à partir d’une valeur seuil.
A partir du calcul des histogrammes peau et non peau nous avons calculé le nombre total des pixels peau et le nombre total des pixels non peau.
Procédé: Pour chaque images donnée, on prend également le masque correspondant et on parcours les pixels de l’image et de son masque tel que pour chaque pixel du masque supérieur à la valeur seuil, le pixel est classé peau, pour chaque valeur de pixel du masque inférieur à la valeur seuil, le pixel est classé non peau.
# Étape 6: Détection de la peau dans les images
A cette étape, plusieurs méthodes sont possibles pour détecter la peau dans l’image. Nous avons choisi la méthode de Classifieur Bayésien. Cette méthode tient en compte les résultats que nous avons obtenus de nos histogrammes précédemment. En effet, on détermine à partir de l’histogramme la probabilité qu’un pixel soit peau ou non peau en fonction de sa couleur. Ensuite on utilise la formule suivant pour la détection:
𝑝(𝑝𝑒𝑎𝑢|𝑐) = 𝑝(𝑐|𝑝𝑒𝑎𝑢)𝑝(𝑝𝑒𝑎𝑢)/(𝑝(𝑐|𝑝𝑒𝑎𝑢)𝑝(𝑝𝑒𝑎𝑢) + 𝑝(𝑐|¬𝑝𝑒𝑎𝑢)𝑝(¬𝑝𝑒𝑎𝑢))
𝑝(¬𝑝𝑒𝑎𝑢|𝑐) = 𝑝(𝑐|¬𝑝𝑒𝑎𝑢)𝑝(¬𝑝𝑒𝑎𝑢)/(𝑝(𝑐|¬𝑝𝑒𝑎𝑢)𝑝(¬𝑝𝑒𝑎𝑢) + 𝑝(𝑐|𝑝𝑒𝑎𝑢)𝑝(𝑝𝑒𝑎𝑢))
Ces deux probabilités sont comparées par rapport à un seuil s (0 < s < 1). Le pixel est dit pixel peau si la contrainte suivante est vérifiée :
𝑝(𝑠𝑘𝑖𝑛|𝑐)/𝑝(¬𝑠𝑘𝑖𝑛|𝑐)>s
