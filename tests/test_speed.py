import sys
import time

sys.path.append(".")
from app.predictor.ml_predictor import get_predictor

p = get_predictor()
texts = ["This is a test"] * 80

start = time.time()
preds = p.predict(texts)
end = time.time()

print(f"Time for 80 texts: {end - start:.2f} seconds")
