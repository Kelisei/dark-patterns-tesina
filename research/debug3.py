import sys
from pathlib import Path
sys.path.append('C:\\Users\\frank\\Downloads\\Tesina\\dark-patterns-tesina')
from src.predictor.ml_predictor import get_predictor, normalize_placeholders
p = get_predictor()
texts = ['Sí, me voy a hacer cargo de los gastos que surjan\nAgregar protección']
print(p.predict(texts))
print(p.pipeline.predict_proba(texts))
print(p.labels)
