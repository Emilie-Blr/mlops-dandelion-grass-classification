from locust import HttpUser, task, between
from PIL import Image
import io
import random

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        # 1. Générer une image aléatoire en mémoire (pas de fichier sur le disque)
        image = Image.new('RGB', (224, 224), color=(
            random.randint(0, 255), 
            random.randint(0, 255), 
            random.randint(0, 255)
        ))
        
        # 2. Convertir l'image en octets (comme si on lisait un fichier)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)

        # 3. Envoyer la requête POST à l'API
        # Le format est : {"file": ("nom_fichier.jpg", données_octets, "type_mime")}
        self.client.post(
            "/predict", 
            files={"file": ("test_image.jpg", img_byte_arr, "image/jpeg")}
        )