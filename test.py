from ultralytics import YOLO
from pathlib import Path
import time

# Cargar el modelo YOLOv5s
model = YOLO('.\\train\\best.pt')

# Ruta al directorio de imágenes de validación
valid_dir = Path('./img')

# Obtener la lista de imágenes de validación
valid_images = list(valid_dir.glob("*.png"))

# Registro de tiempo antes de realizar las inferencias
start_time = time.time()

# Realizar inferencias en las imágenes de validación y guardar
results = model(valid_images, save=True)

# Realizar inferencias en las imágenes de validación sin guardar
# results = model(valid_images)

# Registro de tiempo después de realizar las inferencias
end_time = time.time()

# Calcular la velocidad de inferencia
inference_time = end_time - start_time

# Mostrar la velocidad de inferencia
print(f"Tiempo de inferencia: {inference_time:.2f} ")


