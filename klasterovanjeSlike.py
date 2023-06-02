import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd



# Učitajmo sliku

img = Image.open('image.jpg')
gray_image = img.convert('L')

# Convert the image data to a 2-dimensional array
image_array1 = pd.DataFrame(list(gray_image.getdata()), columns=['Pixel Value'])
	
# Konvertirajmo sliku u niz NumPy-a
#prebacuje svaki pixel u 3d tacku sa x, y, rgb(bojom)
img_array = np.array(img)
print(img_array.shape)
#print("Razmak")

# Izrada niza značajki koji sadrži vrijednosti piksela
#pretvaramo u dvije dimenzije, gdje ce druga da sadrzi 3 dimenzioni element
X = img_array.reshape(-1, 3)
print(X.shape)
# Pokrenimo KMeans algoritam s 16 klastera
''''
klasteri = np.arange(2, 22, 2)
greske = np.arange(0, 10, 1, dtype=np.int64)
print(len(greske))
print(len(klasteri))

for i in range(0, 10):
    greske[i] = KMeans(n_clusters=klasteri[i]).fit(X).inertia_
plt.plot(klasteri, greske)
plt.show()
'''
kmeans = KMeans(n_clusters=8).fit(X)
# Dodijelimo svaki piksel klasteru
labels = kmeans.predict(X)

# Stvorimo novu sliku koristeći nove vrijednosti piksela
new_image = np.zeros_like(X)
for i, label in enumerate(labels):
    new_image[i] = kmeans.cluster_centers_[label]

# Ponovo oblikujemo sliku u originalni oblik
new_image = new_image.reshape(img_array.shape)

# Prikažimo originalnu sliku i novu sliku
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img)
axs[0].set_title('Originalna slika')
axs[1].imshow(new_image.astype('uint8'))
axs[1].set_title('Slika nakon redukcije boja')
plt.show()
