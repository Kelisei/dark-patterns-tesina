from locust import HttpUser, task, between

class ExtUser(HttpUser):
    wait_time = between(0.1, 1)

    @task
    def detect_dark_patterns(self):
        payload = {
            "texts": [
                {"text": "¡Solo quedan 2 lugares! Apúrate o te lo pierdes. Compra ya", "id": "1", "path": "/div"},
                {"text": "No gracias, prefiero perder la oferta", "id": "2", "path": "/button"},
                {"text": "123 personas están mirando esto", "id": "3", "path": "/span"}
            ]
        }
        self.client.post("/detect", json=payload)
