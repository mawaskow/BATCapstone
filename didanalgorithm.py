# import statements
from PIL import Image # PIL library named pillow
import numpy as np
from skimage import data # from scikit library
from skimage.feature import match_template
import matplotlib.pyplot as plt

###################################
### Load all project images
###################################

print("Loading image file as array...")
ImagenTotal = np.asarray(Image.open('../Tutorial/Tozeur/Chabbat.png'))
#multiplesizes: small, medium, large or extra

print("Loading template files as arrays...")
ImagenTemplateSmall = np.asarray(Image.open('../Tutorial/Tozeur/Template1.png'))
ImagenTemplateMedium = np.asarray(Image.open('../Tutorial/Tozeur/Template2.png'))
ImagenTemplateLarge = np.asarray(Image.open('../Tutorial/Tozeur/Template3.png'))
ImagenTemplateExtra = np.asarray(Image.open('../Tutorial/Tozeur/Template4.png'))

###################################
### Display project images
###################################

print("Selecting 2D arrays from image arrays...")
#notice that we work with only one band
imagen = ImagenTotal[:,:,1]
SmallTrees =ImagenTemplateSmall[:,:,2]
MedTrees = ImagenTemplateMedium[:,:,2]
LrgTrees = ImagenTemplateLarge[:,:,2]
ExLrgTrees = ImagenTemplateExtra[:,:,2]
#print(arbol)

print("Building template image display...")
fig = plt.figure(figsize=(15, 4))
ax1 = plt.subplot(1, 4, 1)
ax2 = plt.subplot(1, 4, 2,sharex=ax1,sharey=ax1)
ax3 = plt.subplot(1, 4, 3,sharex=ax1,sharey=ax1)
ax4 = plt.subplot(1, 4, 4,sharex=ax1,sharey=ax1)

ax1.imshow(SmallTrees, cmap=plt.cm.RdYlGn)  #Apply whatver color you wish  summer
ax1.set_title('Type-1')

ax2.imshow(MedTrees, cmap=plt.cm.RdYlGn)   #Apply whatver color you wish  YlGn
ax2.set_title('Type-2')

ax3.imshow(LrgTrees, cmap=plt.cm.RdYlGn)  #Apply whatver color you wish  Greens
ax3.set_title('Type-3')

ax4.imshow(ExLrgTrees, cmap=plt.cm.RdYlGn)  #Apply whatver color you wish 
ax4.set_title('Type-4')

plt.savefig('./Outputs/didan_alg_templatedisp.png', dpi=300, bbox_inches='tight')

###################################
### Apply the shape/pattern matching using a threshold of
###################################

#creating results for every tree type
# The > 0.6 in this case is the tolerance 

# Adjust the threshold to eliminate error and noise 

print("Searching for template matches...")
tol = 0.70
#small
resultsmall = match_template(imagen, SmallTrees)
resultsmallquery = np.where(resultsmall>tol)
#medium
resultmedium = match_template(imagen, MedTrees)
resultmediumquery = np.where(resultmedium>tol)
#large
resultlarge = match_template(imagen, LrgTrees)
resultlargequery = np.where(resultlarge>tol)
#extra
resultextra = match_template(imagen, ExLrgTrees)
resultextraquery = np.where(resultextra>tol)

###################################
### Organize results for display
###################################

def listapuntos(result):
    xlist = []
    ylist = []
    for punto in range(np.shape(result)[1]):
        xlist.append(result[1][punto])
        ylist.append(result[0][punto])
    return xlist, ylist

print("Building results display...")
#show the interpreted results 
plt.figure(figsize =(20,10))
#small
plt.plot(listapuntos(resultsmallquery)[0], listapuntos(resultsmallquery)[1], 'o', 
         markeredgecolor='g', markerfacecolor='none', markersize=5, label="Palm Tree - Type 1")
#medium
plt.plot(listapuntos(resultmediumquery)[0], listapuntos(resultmediumquery)[1], 'o', 
         markeredgecolor='r', markerfacecolor='none', markersize=5, label="Palm Tree - Type 2")
#large
plt.plot(listapuntos(resultlargequery)[0], listapuntos(resultlargequery)[1], 'o', 
         markeredgecolor='b', markerfacecolor='none', markersize=5, label="Palm Tree - Type 3")
#extra
plt.plot(listapuntos(resultextraquery)[0], listapuntos(resultextraquery)[1], 'o', 
         markeredgecolor='y', markerfacecolor='none', markersize=5, label="Palm Tree - Type 4")
plt.imshow(ImagenTotal[10:-10,10:-10,:])
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('./Outputs/didan_alg_tree_class.png', dpi=300, bbox_inches='tight')
plt.show()

print("Building classification library...")
result_types = [resultsmallquery, resultmediumquery, resultlargequery, resultextraquery]
key_nms = ["small", "medium", "large", "extra"]
count_dct = {}
for i in range(len(result_types)):
    x,y = listapuntos(result_types[i])
    count_dct[key_nms[i]] = len(x)
total = 0
for i in count_dct.keys():
    total += count_dct[i]
count_dct["total"] = total
print(count_dct)
print("Program ended.")