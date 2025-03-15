from datasets import load_dataset

# Cargar el conjunto de datos
ds = load_dataset("dwb2023/brain-tumor-image-dataset-semantic-segmentation", cache_dir=r"C:\github\res-u-net\dataset")

# Mostrar la estructura del conjunto de datos
print(ds)

# Acceder a una muestra de datos
sample = ds['train'][0]
print(sample)

# Mostrar la ruta de la imagen
print(sample['image'])