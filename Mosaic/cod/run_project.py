# Parametrii algoritmului sunt definiti in clasa Parameters.
from cod.parameters import *
from cod.build_mosaic import *

# numele imaginii care va fi transformata in mozaic
#load_pieces_cifar("./../cifar-10-batches-py", "bird")
image_path = './../data/imaginiTest/obama.jpeg'
params = Parameters(image_path, gray=True)

# directorul cu imagini folosite pentru realizarea mozaicului
params.small_images_dir = './../data/colectie'
# tipul imaginilor din director
params.image_type = 'png'
# numarul de piese ale mozaicului pe orizontala
# pe verticala vor fi calcultate dinamic a.i sa se pastreze raportul
params.num_pieces_horizontal = 100
# afiseaza piesele de mozaic dupa citirea lor
params.show_small_images = False
# modul de aranjarea a pieselor mozaicului
# optiuni: 'aleator', 'caroiaj'
params.layout = 'caroiaj'
# criteriul dupa care se realizeaza mozaicul
# optiuni: 'aleator', 'distantaCuloareMedie'
params.criterion = 'distantaCuloareMedie'
# daca params.layout == 'caroiaj', sa se foloseasca piese hexagonale
params.hexagon = True
params.neigh = True

img_mosaic = build_mosaic(params)
cv.imwrite('obama.png', img_mosaic)


