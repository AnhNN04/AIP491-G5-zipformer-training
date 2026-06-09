from src.usecases.dialect_routing import DialectRouter

def test_dialect_router_empty_text():
    router = DialectRouter()
    res = router.classify_dialect("")
    assert res["inferred"] == "NORTHERN"
    assert res["probability"] == 0.70

def test_dialect_router_no_markers():
    router = DialectRouter()
    res = router.classify_dialect("xin chào mọi người hôm nay trời rất đẹp")
    assert res["inferred"] == "NORTHERN"
    assert res["probability"] == 0.70

def test_dialect_router_southern_winner():
    router = DialectRouter()
    # "má", "vô", "nè" are Southern indicators
    res = router.classify_dialect("má ơi vô ăn cơm nè")
    assert res["inferred"] == "SOUTHERN"
    assert res["probability"] > 0.90

def test_dialect_router_central_winner():
    router = DialectRouter()
    # "chi", "răng", "tui" are Central indicators
    res = router.classify_dialect("răng chi rứa tui không biết mô")
    assert res["inferred"] == "CENTRAL"
    assert res["probability"] > 0.90

def test_dialect_router_northern_winner():
    router = DialectRouter()
    # "bố", "nhé", "quả" are Northern indicators
    res = router.classify_dialect("bố ơi ăn quả này nhé")
    assert res["inferred"] == "NORTHERN"
    assert res["probability"] > 0.90

def test_dialect_router_mixed_inputs():
    router = DialectRouter()
    # 2 Southern ("má", "lẹ") vs 1 Northern ("nhé")
    res = router.classify_dialect("má ơi đi lẹ lên nhé")
    assert res["inferred"] == "SOUTHERN"
    # winner_count = 2, total_regional_hits = 3 -> p = 0.6 + 0.38*(2/3) = 0.85
    assert res["probability"] == 0.85
