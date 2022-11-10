import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import copy




##############################
# TRIANGULACIÓN DE POLÍGONOS #
##############################


#Función que detecta cuando un punto o puntos están en un triángulo.
def in_triangle(points, triangle):
    t1, t2, t3 = triangle
    p = points.reshape(-1, 2)
    ret = []
    
    #Tomar 2 pares de vértices (ta, tb) y ver si point y el otro vértice tr están al mismo lado de la recta que forma ta, tb
    for ta, tb, tr in [(t1, t2, t3), (t1, t3, t2), (t2, t3, t1)]:
        ortog = (tb-ta) @ np.array([[0, -1], [1, 0]])
        ret.append(((tr - 0.5*(ta+tb)) @ ortog) * ((points - 0.5*(ta+tb)) @ ortog) >= 0)
    return np.all(ret, axis=0)

#Toma 3 puntos secuenciales y mediante producto punto evalúa el ángulo de giro (con signo) entre el vector p2-p1 y p3-p2
def turn(triang):
    v1 = triang[1]-triang[0]
    v2 = triang[2]-triang[1]
    ang = np.arccos(v1@v2/(np.linalg.norm(v1)*np.linalg.norm(v2)))/(2*np.pi)
    sign = v1[0]*v2[1] - v1[1]*v2[0]
    return ang*sign/abs(sign)

#Método de triangulación para determinar si un punto o puntos están en el polígono
def in_polygon(points, polygon, verbose=False):
    #Order representa los puntos que actualmente están en el polígono, los cuales se irán descartando en un proceso iterativo
    order = list(range(len(polygon)))
    
    #La suma de los ángulos de giro de todo polígono sin intersecciones es 2pi o -2pi. Lo calculamos para detectar concavidad
    angle = sum(turn(polygon[(order*2)[t: t+3]]) for t in range(len(order)))
    ret = []
    while len(order)>= 3:
        print(f"Computando triángulo {len(polygon)-len(order)+1} de {len(polygon)-2}") if verbose else None
        nowlen=len(order) #Para evitar loops infinitos, se arroja un error cuando no se descarta un punto en un ciclo
        
        for t in range(len(order)):
            #En un ciclo se evalúan todos los triángulos formados por 3 vértices consecutivos
            tri = polygon[(order*2)[t: t+3]]
            
            #El triángulo es válido si el ángulo de giro es del mismo signo que el del polígono y si no contiene otros vértices
            if turn(tri)*angle > 0 and np.sum(in_triangle(polygon[(2*order)[t+3:t+len(order)]], tri))==0:
                #Se evalúan los puntos si pertenecen al triángulo, se descarta el vértice central y se repite el ciclo
                ret.append(in_triangle(points, tri))
                order.remove((2*order)[t+1])
                break
        if len(order)==nowlen: 
            print(polygon[order])
            raise Exception("Error en el ciclo!")
    
    #Puntos que pertenecen a cualquiera de los triángulos son considerados dentro del polígono
    return np.any(ret, 0)
            
def matrix_polygon(shape, extent, polygon, verbose=False, lower_up=False):
    #Normalizo minmax para evitar en lo posible errores numéricos cuando los triángulos tienen formas muy bizarras
    mean, var = np.mean(extent, axis=0), 100/(np.max(extent, axis=0)- np.min(extent, axis=0))
    extent, polygon = (var*(np.array(extent)-mean)), var*(np.array(polygon)-mean)
    
    def midspace(c1, c2, size):
        lins = np.linspace(c1, c2, size+1)
        return np.mean((lins[1:], lins[:-1]), axis=0)
    side1 = midspace(extent[0], extent[2], shape[0])
    side2 = midspace(extent[1], extent[3], shape[0])
    points = np.concatenate([midspace(s1, s2, shape[1]) for s1, s2 in zip(side1, side2)])
    matrix = in_polygon(points, polygon, verbose=verbose)
    return matrix.reshape(shape)
	
	

#################
# VISUALIZACIÓN #
#################

# Función en desuso: Permite proyectar una matriz a un extent no rectangular. Usada para experimentar para construir la clase mapper
def project_trapezoid(corners, shape):
    def midspace(c1, c2, size):
        lins = np.linspace(c1, c2, size+1)
        return np.mean((lins[1:], lins[:-1]), axis=0)
    side1 = midspace(corners[0], corners[1], shape[1])
    side2 = midspace(corners[2], corners[3], shape[1])
    return np.concatenate([midspace(s1, s2, shape[0]) for s1, s2 in zip(side1, side2)])
    
	
#Clase para visualizar mapas en conjunto con polígonos en el mismo extent.
#Es una clase en vez de función porque precomputa la transformación a realizar para mostrar imágenes no rectangulares al crear la clase.
class mapper:
    def __init__(self, extent, shape):
        self.extent=extent
        self.shape = shape
        self.rect = False
        if extent[0,0]-extent[2,0] == extent[1,0]-extent[3,0] and extent[0,0]-extent[1,0] == extent[2,0]-extent[3,0]:
            self.rect = True
            self.extent = np.array([[extent[0,0], extent[3,0]], [extent[0,1], extent[1,1]]])
        else:
            def midspace(c1, c2, size):
                lins = np.linspace(c1, c2, size+1)
                return np.mean((lins[1:], lins[:-1]), axis=0)
            side1 = midspace(extent[0], extent[2], shape[0])
            side2 = midspace(extent[1], extent[3], shape[0])
            points = np.concatenate([midspace(s1, s2, shape[1]) for s1, s2 in zip(side1, side2)])
            new_extent = np.array([[np.max(extent[:,0]), np.min(extent[:,0])], [np.min(extent[:,1]), np.max(extent[:,1])]])
            step_0 = (new_extent[0, 1] - new_extent[0, 0])/shape[0]
            step_1 = (new_extent[1, 1] - new_extent[1, 0])/shape[1]

            cords = np.column_stack(((points[:,0] - new_extent[0,0])//step_0, 
                                     (points[:,1] - new_extent[1,0])//step_1))
            all_cords = np.array([cords] + [cords + np.array([i, j]) for i in range(-1,2) for j in range(-1,2) if (i, j) != (0,0)])

            weights = np.exp(- np.linalg.norm((points - np.array([new_extent[0, 0], new_extent[1, 0]]) 
                             - np.array([step_0, step_1])/2)/np.array([step_0, step_1]) - all_cords  , axis=2)**2).T.flatten()

            cords = (np.clip(all_cords[:, :, 0], 0, shape[0]-1)*shape[1] + np.clip(all_cords[:, :, 1], 
                                                                                   0, shape[1]-1)).T.flatten().astype(int)
            index = np.repeat(np.arange(points.shape[0]), 9).astype(int)
            weight_order = np.argsort(weights)

            self.index  = index[weight_order]
            self.cords = cords[weight_order]
            self.extent = new_extent

    def project(self, mp):
        if self.rect:
            return mp
        else:
            ret = np.zeros(self.shape[0] * self.shape[1])
            ret[self.cords] = mp.flatten()[self.index]
            return ret.reshape(self.shape)
    def draw(self, maps=[], glaciares=[], cmaps="viridis", linewidth=1.5, linecol="white", figsize=(8, 8),
             subset=[[0,1], [0,1]], center_scale = False, colorbar=False, force_extent = False):
        maps, glaciares, cmaps, center_scale = [v if type(v) == list else [v] for v in [maps, glaciares, cmaps, center_scale]]
        maps = [self.project(mp) for mp in maps]
        subset = (np.array(self.shape).reshape(-1, 1)*np.array(subset)).astype(int)
        imshow_extent = self.extent[:,:1]  +  (self.extent[:,1:]-self.extent[:,:1]) * subset/np.array(self.shape).reshape(-1, 1)
        imshow_extent = np.concatenate((imshow_extent[1], np.flip(imshow_extent[0])))
        glaciares = [np.concatenate((glaciar, [glaciar[0]])) for glaciar in glaciares]
        plt.figure(figsize=figsize)
        for i, mp in enumerate(maps):
            img = mp[subset[0][0]:subset[0][1], subset[1][0]:subset[1][1]]
            if center_scale[i % len(center_scale)]:
                img = np.where(img>0, img/np.max(img), -img/np.min(img))
            plt.imshow(img, extent = imshow_extent, cmap = cmaps[i % len(cmaps)])
            if colorbar is i:
                plt.colorbar()
        for glaciar in glaciares:
            plt.plot(glaciar[:,1], glaciar[:,0], color=linecol, linewidth=linewidth)
        if force_extent:
            plt.xlim(self.extent[1])
            plt.ylim(self.extent[0,[1,0]])
        plt.show()
	
	
def create_colormap(colors, docks, name=""):
    cmap = np.zeros((256, 4))
    points = np.linspace(0, 1, 256)
    colors = np.array(colors)
    for i in range(1, len(colors)):
        condition = np.all((points >= docks[i-1], points<= docks[i]), axis=0)
        cmap[condition] =  colors[i-1] + (points[condition]-docks[i-1]).reshape(-1, 1) * (colors[i]- colors[i-1])
    return LinearSegmentedColormap.from_list(name = name, colors=cmap)
	

#############################
# Exploratory Data Analysis #
#############################

# Para cada punto P se calcula sum(P - QP) donde Q son los puntos contiguos
def substract_correlation(matrix, shape):
    X, Y = matrix.shape
    I, J = shape
    ret = []
    for i in range(-I, I+1):
        for j in range(-J, J+1):
            ret.append( (matrix[I:-I,J:-J]-matrix[I+i:X-I+i,J+j:Y-J+j]) * matrix[I:-I,J:-J] )
    corrs = np.sum(np.array(ret), axis=0)
    return corrs
	
#KDE para calcular la densidad de un vector X
def density(X, data, sigma_scale = .1, cap = 10000):
    sigma = np.std(data) * sigma_scale
    data = np.random.choice(data, cap, replace=False) if len(data)>cap else data
    return np.mean(np.exp(-(X - data.reshape(-1, 1))**2 /(2*sigma**2))/(sigma*np.sqrt(2*np.pi)), axis=0)

# Computa las correlaciones entre pixeles cercanos P y Q donde Q es P desfasado en i, j. Luego promedia las correlaciones para los mismos i y j.
def convolution_hist(matrix, ks=1, cov=False, norm=False, self=False, consider= None):
    ret = []
    I, J = matrix.shape
    if consider is None:
        consider = np.full(matrix.shape, True)

    consider = consider[ks:-ks, ks:-ks].flatten()
    
    for i in range(-ks, ks+1):
        for j in range(-ks, ks+1):
            if (i, j) == (0, 0) and self:
                ret.append(matrix[ks:-ks, ks:-ks].flatten()[consider])
            else:
                ret.append((matrix[ks:-ks, ks:-ks]-matrix[ks+i:I-ks+i, ks+j:J-ks+j]).flatten()[consider])

    if cov:
        mean = np.mean(matrix) if norm else 0
        for i in range(-ks, ks+1):
            for j in range(-ks, ks+1):
                ret.append(((matrix[ks:-ks, ks:-ks]-mean) * (matrix[ks+i:I-ks+i, ks+j:J-ks+j] - mean)).flatten()[consider])
    
    return ret

# Divide el tamaño por 2 en cada dimensión promediando.
def avg_pool(matrix):
    boolean = type(matrix[0,0]) == np.bool_
    mat = matrix*1
    for _ in range(2):
        mat = np.mean(mat[:, :2* (mat.shape[1]//2)].reshape(mat.shape[0], mat.shape[1]//2, 2) , axis=2).T
    if boolean:
        mat = mat>0.5
    return mat

# Calcula las correlaciones del pixel central con los pixeles cercanos y lo visualiza suavizando radialmente.
def circular_hist(hists, size=24, rang=.01, sigma_scale=.1):
    msize = 2*size +1
    matrix = np.zeros((msize, msize))
    weights = np.zeros((msize, msize))
    coord = np.tile([np.arange(msize)-size], msize).reshape((msize, msize))
    coord = np.array((coord, coord.T)).T
    norm = np.linalg.norm(coord, axis=2)

    for i in range(-1, 2):
        for j in range(-1,2):
            if not (i, j) == (0, 0):
                my_vec = np.array([size*i, size*j])/np.sqrt(i**2+j**2)
                
                #Coseno del angulo entre el las coordenadas de la matriz y el vector
                weight = np.sum(coord*my_vec, axis=2)/(np.where(norm==0, 1, norm)*np.linalg.norm(my_vec))
                #Ángulo / pi. Se le aplica 2*abs() -1 para que tanto el ángulo 0 y 1 tengan peso 1 y el ángulo 0.5 tenga peso 0.
                weight = np.abs(1-2*np.arccos(np.clip(weight, -1,1))/np.pi)**2
                weight[size, size]=1
                X = np.flip(rang * np.sum(coord*my_vec, axis=2)/(np.linalg.norm(my_vec)**2))

                dens = density(X.flatten(), hists[3*i+j+4], sigma_scale=sigma_scale).reshape(matrix.shape)

                matrix+= dens * weight
                weights+= weight
    return matrix/weights
	
	
#################################
# Modelo de Clasificación Dummy #
#################################

# Modelo dummy que clasifica puntos según están o no en un glaciar. Para clasificar toma 1.- Su velocidad, 2.- Las diferencias con los pixeles cercanos 3.- Las covarianzas. Todos son opcionales.
class conv_logreg():
    def __init__(self, ks=1, nlayers=3, itself=True, dif=True, cov=False):
        self.itself = itself
        self.dif =dif
        self.cov = cov
        self.nlayers = nlayers
        self.ks = ks if type(ks) in (tuple, list, np.ndarray) else [ks]
        self.k = self.ks[(len(self.ks) - self.nlayers) % len(self.ks)]
        self.next_layer = None
    def get_features(self, matrix):
        ks = self.k
        pixels = []
        matrix = np.vstack((np.tile(matrix[0], ks).reshape((ks, -1)), matrix, np.tile(matrix[-1], ks).reshape((ks, -1))))
        matrix = np.hstack((np.tile(matrix[:,0], ks).reshape((-1, ks)), matrix, np.tile(matrix[:,-1], ks).reshape((-1, ks))))
        I, J = matrix.shape
        
        if self.itself:
            pixels.append(matrix[ks:-ks, ks:-ks].flatten())
        for i in range(-ks, ks+1):
            for j in range(-ks, ks+1):
                if self.dif and (i, j) != (0, 0):
                    pixels.append((matrix[ks:-ks, ks:-ks] - matrix[ks+i:I-ks+i, ks+j:J-ks+j]).flatten())
                if self.cov:
                    pixels.append((matrix[ks:-ks, ks:-ks] * matrix[ks+i:I-ks+i, ks+j:J-ks+j]).flatten())
        return np.array(pixels).T
    
    def fit(self, matrix, target, drop_matrix = None, verbose=True):
        self.drop_matrix = drop_matrix	
        self.target = target.flatten()
        self.pixels = self.get_features(matrix)
		
        self.target_train = copy.deepcopy(self.target) if drop_matrix is None else self.target[drop_matrix.flatten()]
        self.pixels_train = copy.deepcopy(self.pixels) if drop_matrix is None else self.pixels[drop_matrix.flatten()]
		
        self.scaler = StandardScaler().fit(self.pixels_train)
        self.pixels = self.scaler.transform(self.pixels)
        self.pixels_train = self.scaler.transform(self.pixels_train)

        self.model = LogisticRegression(max_iter = 500, class_weight="balanced")
        self.model.fit(self.pixels_train, self.target_train*1)
		
        self.probas = self.model.predict_proba(self.pixels)[:, 1]
		
        self.probas_train = self.model.predict_proba(self.pixels_train)[:, 1]
        self.probas_mat = self.probas.reshape(matrix.shape)
        
        self.target_mat = self.target.reshape(matrix.shape)

        if verbose:
            self.report()
        
        if self.nlayers > 1:
            self.next_layer = conv_logreg(self.ks, self.nlayers - 1, self.itself, self.dif, self.cov)
            self.next_layer.fit(*self.max_pool())

    def max_pool(self, probas=None, target=None, drop_matrix = None):
        prb, tar, dmx = (self.probas_mat if probas is None else probas), (self.target_mat*1 if target is None else target), (self.drop_matrix if drop_matrix is None else drop_matrix)
        for _ in range(2):
            prb = np.max( prb[:, :2* (prb.shape[1]//2)].reshape(prb.shape[0], prb.shape[1]//2, 2) , axis=2).T
            tar = np.mean(tar[:, :2* (tar.shape[1]//2)].reshape(tar.shape[0], tar.shape[1]//2, 2) , axis=2).T
            if dmx is not None:
                dmx = np.mean(dmx[:, :2* (dmx.shape[1]//2)].reshape(dmx.shape[0], dmx.shape[1]//2, 2)*1 , axis=2).T
        
        return prb, (tar>=.5), (dmx if dmx is None else (dmx>=.5))
        
    def predict(self, matrix, weight=3):
        pixels = self.get_features(matrix)
        pixels = self.scaler.transform(pixels)
        
        probas = self.model.predict_proba(pixels)[:, 1]
        probas_mat = probas.reshape(matrix.shape)
        
        if self.nlayers == 1:
            return probas_mat
        else:
            next_probas = self.next_layer.predict(self.max_pool(probas_mat)[0], weight=weight)
            next_probas = np.repeat(np.tile(next_probas, 2), 2).reshape(2 * np.array(next_probas.shape))
            next_probas = np.vstack((next_probas, next_probas[-1])) if next_probas.shape[0] < matrix.shape[0] else next_probas
            next_probas = np.hstack((next_probas, next_probas[:,-1:])) if next_probas.shape[1] < matrix.shape[1] else next_probas
            return (probas_mat * next_probas**weight) ** (1.0/(weight+1))
        
    def project(self):
        self.pix_t = self.pixels_train[self.target_train]
        self.pix_f = self.pixels_train[np.logical_not(self.target_train)]
        
        self.pix_t = self.scaler.transform(self.pix_t)
        self.pix_f = self.scaler.transform(self.pix_f)

        self.pca = PCA(n_components=1).fit(self.pixels_train)
        self.pixels_pc = pca.transform(self.pixels_train)

        self.pix_t_pc = np.column_stack((np.sum(self.pix_t * self.model.coef_, axis=1) + self.model.intercept_, 
                                         self.pca.transform(self.pix_t)))
        self.pix_f_pc = np.column_stack((np.sum(self.pix_f * self.model.coef_, axis=1) + self.model.intercept_,
                                         self.pca.transform(self.pix_f)))
        
    
    def report(self):
        print("Remaining layers:", self.nlayers-1)
        prds = self.probas_train>.5
        print("Accuracy:", np.sum(prds == self.target_train)/len(prds))
        print("True positives:", np.sum(prds[self.target_train == 1])/np.sum(self.target_train))
        print("False positives:", np.sum(prds[self.target_train == 0])/np.sum(1-self.target_train))
		
###########################################################################
# Muestreo de Imágenes Satelitales Para Entrenamiento de Redes Neuronales #
###########################################################################

# Permite hacer un mask más ancho. Por defecto, a lo que le aumenta el ancho es a los valores False.
def broaden_mask(mask, thick):
    new_mask = mask.copy()
    I, J = mask.shape
    T = thick
    new_mask[:T] = np.full_like(mask[:T], False)
    new_mask[-T:] = np.full_like(mask[-T:], False)
    new_mask[:,:T] = np.full_like(mask[:,:T], False)
    new_mask[:,-T:] = np.full_like(mask[:,-T:], False)
    
    def minibroaden(shape):
        i, j = shape
        new_mask[T:-T, T:-T] = np.all((new_mask[T:-T, T:-T], mask[T+i:I-T+i, T+j:J-T+j]), axis=0)
    
    for i in (-T, T):
        for j in range(-T, T):
            minibroaden((i, j))
    for j in (-T, T):
        for i in range(-T+1, T-1):
            minibroaden((i, j))
            
    return new_mask

# Genera muestras en base a una máscara que indica cuales pixeles pueden ser usados como centro. Las muestras son simplemente los índices de inicio y fin de cada cuadrado
# Al muestrear, el peso de los pixeles se divide por decay matrix, cuya intensidad y forma están dadas por weight_decay y power, haciendo que sea menos probable muestrear de esos pixeles.
def generate_samples(mask, size, n_samples, weight_decay = 100, power=1, decay =True):
    samples = np.zeros((n_samples, 4))
    mask_weights, S = mask*1.0, size
    decay_matrix = weight_decay**(np.array([[(size - np.abs(j-size))*(size - np.abs(i-size))/size**2 
                                            for j in range(2*size)] for i in range(2*size)])**power)
    for ns in range(n_samples):
        mask_flat = mask_weights.flatten()
        center_flat = np.random.choice(np.arange(len(mask_flat)), p = mask_flat/np.sum(mask_flat))
        C = center_flat // mask.shape[1], center_flat % mask.shape[1]
        samples[ns] = np.array([C[0]-size//2, C[0]+size//2, C[1]-size//2, C[1]+size//2])
        if decay:
            dm = [max(C[0]-S,0) - C[0], min(C[0]+S, mask.shape[0]) - C[0], max(C[1]-S,0) - C[1], min(C[1]+S, mask.shape[1]) - C[1]]
            mask_weights[C[0] +dm[0]: C[0] +dm[1], C[1] +dm[2]: C[1] +dm[3]]= \
            mask_weights[C[0] +dm[0]: C[0] +dm[1], C[1] +dm[2]: C[1] +dm[3]] / decay_matrix[S +dm[0]: S +dm[1], S +dm[2]: S +dm[3]]

        if ns%50 ==0: print(ns)
        
    return mask_weights, samples
	
# En base al output de generate_samples se obtienen las imágenes asociadas a las coordenadas.
# Se necesita una lista de imágenes desde las que se tomarán muestras, cada una con igual forma. Se pueden realizar rotaciones y flips para aumentar los datos.
def sample_images(samples, mask, images, augmentate=True, weights = None):
    samples = samples.astype(int)
    weights = weights if weights is None else np.array(weights)/np.sum(weights)
    imgs = []
    msks = []
    order = np.arange(len(samples))
    np.random.shuffle(order)
    for sam in samples[order]:
        image = images[np.random.choice(np.arange(len(images)), p = weights)]
        img=image[sam[0]:sam[1], sam[2]:sam[3]]
        msk= mask[sam[0]:sam[1], sam[2]:sam[3]]
        if augmentate:
            rotation = np.random.randint(4)
            transpos = (np.random.randint(2) ==0)
            img = np.rot90(img, rotation)
            msk = np.rot90(msk, rotation)
            img = np.transpose(img, axes = (1, 0, 2)) if transpos else img
            msk = msk.T if transpos else msk
        imgs.append(img)
        msks.append(msk)
    return imgs, msks 






