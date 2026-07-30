from app.predictor.ml_predictor import normalize_placeholders, has_social_proof

def test_normalize_placeholders():
    text1 = "¡Solo quedan 2 horas y 15 minutos!"
    normalized1 = normalize_placeholders(text1)
    assert "quedan __TIMER__" in normalized1.lower() or "__timer__" in normalized1.lower()

    text2 = "El precio es de $ 15.000,50."
    normalized2 = normalize_placeholders(text2)
    assert "__MONEY__" in normalized2

    text3 = "La oferta finaliza el 15/08/2026."
    normalized3 = normalize_placeholders(text3)
    assert "__DATE__" in normalized3

def test_has_social_proof():
    assert has_social_proof("15 personas están viendo esto") == True
    assert has_social_proof("Comprado en más de 20 carritos") == True
    assert has_social_proof("Este producto es muy bueno") == False
