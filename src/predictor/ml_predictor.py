import re
import joblib
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import os

nlp = spacy.load("es_core_news_sm")

# Regex & Matcher Configurations
RE_SPACED_COLON_TIMER = re.compile(r"\b\d{1,2}\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2}){1,3}\b")
RE_D_COLON_TIMER = re.compile(r"\b\d+\s*[dD]\s*:\s*\d{1,2}(?:\s*:\s*\d{1,2}){1,2}\b")
RE_COLON_TIMER = re.compile(r"\b\d{1,2}(?::\d{1,2}){1,4}\b")
RE_UNIT_TIMER = re.compile(
    r"(?ix)\b("
    r"(?:\d+\s*(?:d|días?|dia|day|days))\s*"
    r"(?:\d+\s*(?:h|hs|hr|hrs|horas?))?\s*"
    r"(?:\d+\s*(?:m|min|mins|minutos?))?\s*"
    r"(?:\d+\s*(?:s|seg|segs|segundos?))?"
    r"|"
    r"(?:\d+\s*(?:h|hs|hr|hrs|horas?))\s*"
    r"(?:\d+\s*(?:m|min|mins|minutos?))\s*"
    r"(?:\d+\s*(?:s|seg|segs|segundos?))"
    r")\b"
)
RE_CLOCK_ONLY = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

RE_REMAINING_TIME = re.compile(
    r"(?i)\b(solo\s*)?qued[ae]n?\s*(\d+|__NUM__)\s*(horas?|hs|h|minutos?|min|m|segundos?|seg|s|d[ií]as?)\b"
)

RE_QUEDAN_NUM = re.compile(r"(?i)\b(qued[ae]n?)\s*(\d+|__NUM__)\b")
RE_STOCK_CONTEXT = re.compile(r"(?i)\b(stock|unidades?|disponibles?|restantes?)\b")

RE_LAST_WINDOW = re.compile(
    r"(?i)\b(últim[oa]s?|ultim[oa]s?)\s*(\d+|__NUM__)\s*(horas?|hs|h|d[ií]as?|dia|days?)\b"
)
RE_IN_LAST_WINDOW = re.compile(
    r"(?i)\ben\s*las\s*(últim[oa]s?|ultim[oa]s?)\s*(\d+|__NUM__)\s*(horas?|hs|h|d[ií]as?)\b"
)

RE_DATE = re.compile(r"\b(\d{1,2}[/-]\d{1,2})(?:[/-]\d{2,4})?\b")
RE_PERCENT = re.compile(r"\b\d{1,3}\s*%\b")
RE_CURRENCY = re.compile(r"(?i)(?:ars|\$|usd|u\$s|€)\s*\d+(?:[.,]\d+)*\b")
RE_STANDALONE_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")

RE_SOCIAL_UNITS = re.compile(
    r"(?i)\b(?:__NUM__|__PEOPLE__)\s*(comprados?|vendidos?|pedidos?|ventas?|visit(as|os)?|vistas?|"
    r"mirando|viendo|en\s*carritos?|añadid[oa]s?|agregad[oa]s?)\b"
)

RE_SOCIAL_PROOF = re.compile(
    r"(?ix)\b("
    r"__social_count__"
    r"|(?:\d+|__num__|__people__)\s*(?:personas?)\s*(?:est[aá]n\s*)?(?:viendo|mirando)\b"
    r"|en\s*(?:m[aá]s\s*de\s*)?(?:\d+|__num__|__people__)\s*carritos?\b"
    r"|acaba\s*de\s*comprar\b"
    r"|(?:vendid[oa]s?|comprad[oa]s?|pedidos?)\b"
    r")"
)

URGENCY_TRIGGERS = [
    "apurate",
    "apúrate",
    "ya",
    "no te lo pierdas",
    "última",
    "ultima",
    "oportunidad",
    "comprá",
    "compra",
    "reservá",
    "reserva",
    "oferta",
    "últimas",
    "ultimas",
    "flash",
    "sale",
    "relámpago",
    "relampago",
    "aprovecha",
    "ahora o nunca",
    "por tiempo limitado",
    "ultimo día",
    "último día",
    "última oportunidad",
    "ultima oportunidad",
    "solo hoy",
    "sólo hoy",
    "solo ahora",
    "sólo ahora",
    "termina en",
    "finaliza en",
    "quedan",
    "queda",
    "últimos",
    "ultimos",
    "stock bajo",
    "casi agotado",
    "__timer__",
    "__stock__",
]

TECH_NOUNS = [
    "sesión",
    "sesion",
    "batería",
    "bateria",
    "dispositivo",
    "equipo",
    "sistema",
    "conexión",
    "conexion",
    "proceso",
    "operación",
    "operacion",
    "pantalla",
    "aplicación",
    "aplicacion",
    "instancia",
    "entorno",
    "pedido",
    "token",
]

END_VERBS = [
    "expira",
    "expirar",
    "caduca",
    "caducar",
    "vence",
    "vencer",
    "cierra",
    "cerrar",
    "finaliza",
    "finalizar",
    "termina",
    "terminar",
    "se agotará",
    "se agotara",
    "agotarse",
    "apaga",
    "apagarse",
    "desconecta",
    "desconectarse",
    "bloquea",
    "bloquearse",
]

EVENT_TERMS = ["clase", "evento", "live", "streaming", "stream", "partido", "examen"]
EVENT_START_VERBS = [
    "empieza",
    "comienza",
    "inicia",
    "arranca",
    "comenzar",
    "empezar",
    "iniciar",
]

re_in_time = re.compile(
    r"\b\d+\s*(?:segundos?|minutos?|horas?|hs|h|m|s|d[ií]as?|d[ií]a)\b", re.IGNORECASE
)
re_clock = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

matcher = Matcher(nlp.vocab)
shaming_matcher = Matcher(nlp.vocab)
phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

shaming_matcher.add(
    "FP_VERB",
    [[{"POS": "VERB", "MORPH": {"IS_SUPERSET": ["Person=1", "Number=Sing"]}}]],
)
shaming_matcher.add(
    "FP_COPULA",
    [
        [
            {
                "DEP": "cop",
                "POS": "AUX",
                "MORPH": {"IS_SUPERSET": ["Person=1", "Number=Sing"]},
            }
        ]
    ],
)
shaming_matcher.add(
    "FP_ME_VERB",
    [
        [
            {"POS": "PRON", "MORPH": {"IS_SUPERSET": ["Person=1", "Number=Sing"]}},
            {"POS": "VERB"},
        ]
    ],
)

matcher.add(
    "METADATA_UNITS_FULL",
    [
        [
            {"IS_SPACE": True, "OP": "*"},
            {"LIKE_NUM": True},
            {"IS_SPACE": True, "OP": "*"},
            {
                "LOWER": {
                    "IN": [
                        "colores",
                        "color",
                        "tamaños",
                        "tamaño",
                        "talles",
                        "talle",
                        "piezas",
                        "pieza",
                        "unidades",
                        "unidad",
                    ]
                }
            },
            {"IS_SPACE": True, "OP": "*"},
        ]
    ],
)
matcher.add(
    "NEUTRAL_NO_THANKS_FULL",
    [
        [
            {"IS_SPACE": True, "OP": "*"},
            {"LOWER": "no"},
            {"IS_SPACE": True, "OP": "*"},
            {"LOWER": "gracias"},
            {"IS_SPACE": True, "OP": "*"},
        ]
    ],
)
matcher.add(
    "LAUNCH_AVAILABLE_SOON",
    [
        [
            {"IS_SPACE": True, "OP": "*"},
            {"LOWER": "disponible"},
            {"IS_SPACE": True, "OP": "*"},
            {"LOWER": {"IN": ["próximamente", "proximamente"]}},
            {"IS_SPACE": True, "OP": "*"},
        ]
    ],
)

SAFE_FILTER_TERMS = [
    "filtro",
    "filtros",
    "cerca del centro",
    "centro",
    "sauna",
    "piscina",
    "desayuno",
    "wifi",
    "estacionamiento",
    "ordenar por",
    "distancia",
    "ubicación",
    "ubicacion",
    "categoría",
    "categoria",
    "estrellas",
]
phrase_matcher.add("SAFE_FILTERS", [nlp.make_doc(t) for t in SAFE_FILTER_TERMS])

SAFE_UI_TERMS = [
    "total", "subtotal", "impuesto", "impuestos", "tarifa", "precio",
    "código", "cvv", "código (cvv)", "tarjeta", "tarjetas", 
    "número de tarjeta", "vencimiento", "débito", "crédito",
    "no seleccionada", "seleccionar", "seleccione", "selección",
    "opción", "opciones", "método de pago", "pago", "correo", 
    "email", "contraseña", "usuario", "login", "ingresar", 
    "registrarse", "nombre", "apellido", "dni", "documento", "teléfono"
]

SAFE_FEATURE_TERMS = [
    "cobertura",
    "asistencia",
    "millas ilimitadas",
    "millaje",
    "seguro",
    "neumáticos",
]

# Detecta strings exactos como "__MONEY__ / día" o "__MONEY__ por mes"
RE_SAFE_RATE = re.compile(
    r"^\s*__MONEY__\s*(?:/|por)\s*(d[ií]as?|mes|años?|horas?|hs|h)\s*$", re.IGNORECASE
)
phrase_matcher.add("URGENCY_TRIGGERS", [nlp.make_doc(t) for t in URGENCY_TRIGGERS])
phrase_matcher.add("TECH_NOUNS", [nlp.make_doc(t) for t in TECH_NOUNS])
phrase_matcher.add("END_VERBS", [nlp.make_doc(t) for t in END_VERBS])
phrase_matcher.add("EVENT_TERMS", [nlp.make_doc(t) for t in EVENT_TERMS])
phrase_matcher.add("EVENT_START", [nlp.make_doc(t) for t in EVENT_START_VERBS])
phrase_matcher.add("SAFE_UI", [nlp.make_doc(t) for t in SAFE_UI_TERMS])
phrase_matcher.add("SAFE_FEATURES", [nlp.make_doc(t) for t in SAFE_FEATURE_TERMS])


def normalize_placeholders(
    text: str, normalize_stock=True, normalize_people=True
) -> str:
    t = str(text)

    t = RE_SPACED_COLON_TIMER.sub("__TIMER__", t)
    t = RE_D_COLON_TIMER.sub("__TIMER__", t)
    t = RE_COLON_TIMER.sub("__TIMER__", t)
    t = RE_UNIT_TIMER.sub("__TIMER__", t)
    t = RE_CLOCK_ONLY.sub("__TIMER__", t)

    t = RE_REMAINING_TIME.sub("quedan __TIMER__", t)

    if normalize_people:
        t = re.sub(r"(?i)\b\d+\s*personas?\b", "__PEOPLE__", t)
        t = re.sub(
            r"(?i)\ben\s*m[aá]s\s*de\s*\d+\s*carritos\b", "en __PEOPLE__ carritos", t
        )
        t = re.sub(r"(?i)\ben\s*\d+\s*carritos\b", "en __PEOPLE__ carritos", t)

    t = RE_DATE.sub("__DATE__", t)
    t = RE_PERCENT.sub("__PCT__", t)
    t = RE_CURRENCY.sub("__MONEY__", t)
    t = RE_STANDALONE_NUM.sub("__NUM__", t)

    t = RE_LAST_WINDOW.sub("__LAST_WINDOW__", t)
    t = RE_IN_LAST_WINDOW.sub("en __LAST_WINDOW__", t)

    if normalize_stock:
        t = re.sub(
            r"(?i)\bsolo\s*qued[ae]n?\s*__NUM__\s*en\s*stock\b",
            "Solo quedan __STOCK__ en stock",
            t,
        )
        t = re.sub(
            r"(?i)\(\s*__NUM__\s*disponibles?\s*\)", "(__STOCK__ disponibles)", t
        )
        if RE_STOCK_CONTEXT.search(t):
            t = RE_QUEDAN_NUM.sub(r"\1 __STOCK__", t)

    t = re.sub(
        r"(?i)\b("
        r"casi\s*agotad[oa]s?|"
        r"[uú]ltimas?\s*unidades|"
        r"stock\s*bajo|"
        r"se\s+agota(r[aá]|r[aá]n)?\s+pronto|"
        r"una\s+vez\s+que\s+se\s+agote\s+se\s+acab[oó]|"
        r"vendi[eé]ndose\s+r[aá]pido|"
        r"se\s+vende\s+r[aá]pido|"
        r"movi[eé]ndose\s+r[aá]pido|"
        r"alta\s+demanda|"
        r"en\s+tu\s+carrito\s+se\s+est[aá]\s+agotando"
        r")\b",
        "__SCARCITY__",
        t,
    )

    t = RE_SOCIAL_UNITS.sub("__SOCIAL_COUNT__", t)

    t = re.sub(r"\s{2,}", " ", t).strip()
    return t


def has_social_proof(text: str) -> bool:
    return bool(RE_SOCIAL_PROOF.search(str(text).lower()))


def has_shaming_pattern(doc):
    return len(shaming_matcher(doc)) > 0


def has_urgency_trigger(doc):
    matches = phrase_matcher(doc, as_spans=False)
    return any(nlp.vocab.strings[mid] == "URGENCY_TRIGGERS" for mid, _, _ in matches)


def check_full_text_match(doc, label_prefixes):
    matches = matcher(doc)
    non_space_tokens = [i for i, tok in enumerate(doc) if not tok.is_space]
    if not non_space_tokens:
        return False
    first_token = non_space_tokens[0]
    last_token = non_space_tokens[-1]
    for match_id, start, end in matches:
        lab = nlp.vocab.strings[match_id]
        if any(lab.startswith(p) for p in label_prefixes):
            if start <= first_token and end > last_token:
                return True
    return False


def is_safe_non_pattern(text: str) -> bool:
    if RE_SAFE_RATE.match(text):
        return True
    doc = nlp(text.lower())
    if has_urgency_trigger(doc):
        return False
    if check_full_text_match(doc, ["METADATA_"]):
        return True
    if check_full_text_match(doc, ["NEUTRAL_"]):
        return True
    if check_full_text_match(doc, ["LAUNCH_"]):
        return True
    pm = phrase_matcher(doc, as_spans=False)
    has_safe_ui = any(
        nlp.vocab.strings[mid] in ["SAFE_UI", "SAFE_FEATURES", "SAFE_FILTERS"]
        for mid, _, _ in pm
    )

    if has_safe_ui:
        return True

    return False


def is_anti_dark_fp(text: str) -> bool:
    doc = nlp(text.lower())
    text_lower = text.lower()
    if has_urgency_trigger(doc):
        return False

    pm = phrase_matcher(doc, as_spans=False)
    has_event = any(nlp.vocab.strings[mid] == "EVENT_TERMS" for mid, _, _ in pm)
    has_start = any(nlp.vocab.strings[mid] == "EVENT_START" for mid, _, _ in pm)
    has_tech = any(nlp.vocab.strings[mid] == "TECH_NOUNS" for mid, _, _ in pm)
    has_end = any(nlp.vocab.strings[mid] == "END_VERBS" for mid, _, _ in pm)

    if has_event and has_start:
        return True

    if has_start and " en " in text_lower:
        if re_in_time.search(text_lower) or re_clock.search(text_lower):
            return True

    if has_tech and has_end:
        if (
            re_in_time.search(text_lower)
            or re_clock.search(text_lower)
            or " en " in text_lower
        ):
            return True

    return False


def prefilter_to_none(text: str) -> bool:
    t = str(text)

    if "__social_count__" in t.lower():
        return False

    doc = nlp(t.lower())
    if has_shaming_pattern(doc):
        return False

    if has_social_proof(t):
        return False

    return is_anti_dark_fp(t) or is_safe_non_pattern(t)


class DarkPatternPredictor:
    def __init__(self, model_path="dark_pattern_model.joblib"):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path
        )
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"El modelo no se encontró en {path}. Por favor ejecuta el notebook."
            )

        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.thresholds = artifact["thresholds"]
        self.labels = artifact["labels"]
        if "shaming" in self.labels:
            shaming_idx = self.labels.index("shaming")
            self.thresholds[shaming_idx] = 0.6


    def predict(self, texts, use_prefilter=True):
        if isinstance(texts, str):
            texts = [texts]

        texts_norm = [normalize_placeholders(t) for t in texts]
        proba = self.pipeline.predict_proba(texts_norm)
        pred = (proba >= self.thresholds).astype(int)

        if use_prefilter:
            disc = [prefilter_to_none(tn) for tn in texts_norm]
            for i, d in enumerate(disc):
                if d:
                    pred[i, :] = 0

        # Mapea resultados
        results = []
        for i, p in enumerate(pred):
            detected_labels = [
                self.labels[j] for j, is_detected in enumerate(p) if is_detected
            ]
            results.append(
                {
                    "original_text": texts[i],
                    "normalized_text": texts_norm[i],
                    "detected": len(detected_labels) > 0,
                    "labels": detected_labels,
                }
            )

        return results


predictor_instance = None


def get_predictor():
    global predictor_instance
    if predictor_instance is None:
        predictor_instance = DarkPatternPredictor()
    return predictor_instance
