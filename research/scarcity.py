    from config import NLP
    from spacy.matcher import Matcher

    from research.DarkPatternPredictor import DarkStrategy
    from src.scarcity.types import ScarcityResponseSchema
    import pandas as pd
    scarcity_matcher = Matcher(NLP.vocab)
    scarcity_patterns = [
        # Oraciones del tipo "Últimas 3 unidades" o "ultimo disponible"
        [
            {"LOWER": {"FUZZY": {"IN": ["ultima", "ultimo"]}}},
            {"TEXT": {"REGEX": "^\d*"}, "OP": "?"},
            {"LOWER": {"FUZZY": {"IN": ["unidade", "disponible"]}}},
        ],
        # Oraciones del tipo "Solo quedan 3"
        [
            {"LOWER": {"FUZZY1": "solo"}},
            {"LOWER": {"FUZZY1": "queda"}},
            {"TEXT": {"REGEX": "^\d+"}},
        ],
        # Oraciones del tipo "Últimas unidades disponibles"
        [
            {
                "LEMMA": {"IN": ["último", "ultimo"]},
                "POS": "ADJ",
            },
            {"IS_DIGIT": True, "OP": "?"},
            {"POS": "NOUN"},
            {"POS": "ADJ", "OP": "?"},
        ],
        # Oraciones del tipo "¡Aprovecha! Quedan pocas unidades"
        [
            {"POS": "PUNCT", "OP": "*"},
            {
                "LEMMA": {"IN": ["quedar", "restar"]},
                "POS": "VERB",
            },
            {
                "LEMMA": {"IN": ["poco", "escaso", "limitado"]},
                "POS": {"IN": ["DET", "ADJ"]},
            },
            {
                "LEMMA": {"IN": ["unidad", "existencia", "articulo", "plaza"]},
                "POS": "NOUN",
            },
            {"POS": "PUNCT", "OP": "*"},
        ],
        [
            {"POS": "PUNCT", "OP": "*"},
            {
                "LEMMA": {
                    "IN": ["comprar", "compra", "adquirir", "pedir", "ordenar", "haz"]
                },
                "POS": {"IN": ["VERB", "NOUN"]},
            },
            {"LEMMA": {"IN": ["ya", "ahora"]}, "POS": "ADV", "OP": "?"},
            {"LEMMA": "mismo", "POS": "ADJ", "OP": "?"},
            {"LEMMA": {"IN": ["antes", "previo"]}, "POS": "ADV"},
            {"LEMMA": "de", "POS": "ADP"},
            {"LEMMA": "que", "POS": "SCONJ"},
            {"LEMMA": "él", "POS": "PRON", "OP": "?"},
            {"LEMMA": {"IN": ["acabar", "terminar", "agotar", "acabar_se"]}, "POS": "VERB"},
            {"POS": "PUNCT", "OP": "*"},
        ],
        [
            {"IS_PUNCT": True, "OP": "*"},
            {"LOWER": {"REGEX": "s[oó]lo"}, "OP": "?"},  # Solo / Sólo (opcional)
            {"LIKE_NUM": True},  # 3 / tres
            {
                "LEMMA": {
                    "IN": [
                        "unidad",
                        "pieza",
                        "artículo",
                        "articulo",
                        "existencia",
                        "producto",
                        "plaza",
                        "stock",
                    ]
                },
                "POS": "NOUN",
            },
            {
                "POS": "ADJ",
                "LEMMA": {
                    "IN": ["restante", "disponible", "limitado", "último", "ultimo", "poco"]
                },
                "OP": "?",  # restantes / disponibles (opcional)
            },
            {"IS_PUNCT": True, "OP": "*"},
        ],
        # Solo 3 restantes unidades (orden adjetivo-nombre, por si viene mal redactado)
        [
            {"IS_PUNCT": True, "OP": "*"},
            {"LOWER": {"REGEX": "s[oó]lo"}, "OP": "?"},
            {"LIKE_NUM": True},
            {
                "POS": "ADJ",
                "LEMMA": {
                    "IN": ["restante", "disponible", "limitado", "último", "ultimo", "poco"]
                },
            },
            {
                "LEMMA": {
                    "IN": [
                        "unidad",
                        "pieza",
                        "artículo",
                        "articulo",
                        "existencia",
                        "producto",
                        "plaza",
                        "stock",
                    ]
                },
                "POS": "NOUN",
            },
            {"IS_PUNCT": True, "OP": "*"},
        ],
        # Solo 3 uds restantes!  /  Solo 3 u. disponibles
        [
            {"IS_PUNCT": True, "OP": "*"},
            {"LOWER": {"REGEX": "s[oó]lo"}, "OP": "?"},
            {"LIKE_NUM": True},
            {"LOWER": {"IN": ["u", "u.", "ud", "uds"]}},  # abreviaturas de unidades
            {
                "POS": "ADJ",
                "LEMMA": {
                    "IN": ["restante", "disponible", "limitado", "último", "ultimo", "poco"]
                },
                "OP": "?",
            },
            {"IS_PUNCT": True, "OP": "*"},
        ],
        # 3 unidades en stock / 3 unidades disponibles (sin "Solo")
        [
            {"IS_PUNCT": True, "OP": "*"},
            {"LIKE_NUM": True},
            {
                "LEMMA": {
                    "IN": [
                        "unidad",
                        "pieza",
                        "artículo",
                        "articulo",
                        "existencia",
                        "producto",
                        "plaza",
                        "stock",
                    ]
                },
                "POS": "NOUN",
            },
            {"LOWER": "en", "OP": "?"},
            {"LOWER": "stock", "OP": "?"},
            {
                "POS": "ADJ",
                "LEMMA": {
                    "IN": ["restante", "disponible", "limitado", "último", "ultimo", "poco"]
                },
                "OP": "?",
            },
            {"IS_PUNCT": True, "OP": "*"},
        ],
        [
            {"IS_PUNCT": True, "OP": "*"},
            {"LOWER": {"REGEX": "s[oó]lo"}},
            {"LIKE_NUM": True},
            {
                "POS": "ADJ",
                "LEMMA": {
                    "IN": ["restante", "disponible", "limitado", "último", "ultimo", "poco"]
                },
            },
            {"IS_PUNCT": True, "OP": "*"},
        ],
    ]
    scarcity_matcher.add("fake_scarcity", scarcity_patterns)


    def check_text_scarcity(text):
        """
        """
        doc = NLP(text)
        for _ in scarcity_matcher(doc):
            return True
        return False


    class ScarcityPredictorNLP(DarkStrategy):
        def predict(self,text: str):
            return check_text_scarcity(text)

        def predict_multiple(self,texts: pd.Series):
            predictions = []
            for text in texts:
                predictions.append(check_text_scarcity(text))
            return predictions