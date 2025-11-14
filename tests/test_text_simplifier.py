from app import text_simplifier as ts

OCR_FIXTURE = (
    "令和6年度 受験案内\n"
    "横浜国際高等学校（国際文化科）\n"
    "鬱蒼たる森でAIカタカナを解析。\n"
)


def test_katakana_to_hiragana_basic():
    assert ts.katakana_to_hiragana("スーパーAIカタカナ") == "すーぱーAIかたかな"


def test_text_to_hiragana_on_fixture():
    converted = ts.text_to_hiragana(OCR_FIXTURE)
    assert "れいわ6ねんど" in converted
    assert converted.endswith("をかいせき。\n")
    assert converted.count("\n") == OCR_FIXTURE.count("\n")


def test_limit_to_kyouiku_kanji_on_fixture():
    simplified = ts.limit_to_kyouiku_kanji(OCR_FIXTURE)
    assert "鬱" not in simplified
    assert "うっそうたる" in simplified
    assert "よこはま" in simplified
    assert "国際" in simplified  # stays because both characters are in 教育漢字
