import sys
from pathlib import Path
sys.path.append('C:\\Users\\frank\\Downloads\\Tesina\\dark-patterns-tesina')
from src.predictor.ml_predictor import get_predictor
p = get_predictor()
print(p.thresholds)
