import sys
from pathlib import Path
sys.path.append('C:\\Users\\frank\\Downloads\\Tesina\\dark-patterns-tesina')
from src.predictor.ml_predictor import get_predictor, normalize_placeholders
p = get_predictor()
texts = ['Sí, me voy a hacer cargo de los gastos que surjan', 'Yes, I will take care of any expenses that may be incurred']
text_norm = [normalize_placeholders(t) for t in texts]
print(p.pipeline.predict_proba(text_norm))
print(p.labels)
