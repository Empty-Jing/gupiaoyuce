import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "frontend"))

from theme import LIGHT_THEME, DARK_THEME, get_theme


_SCALAR_KEYS = {
    "bg_primary", "bg_secondary", "text_primary", "text_secondary", "text_tertiary",
    "rise_color", "fall_color", "accent_blue",
    "card_bg", "card_border", "card_shadow",
    "glass_bg", "glass_border",
    "plotly_paper_bg", "plotly_plot_bg", "plotly_font_color",
    "plotly_gridcolor", "plotly_spike_color",
    "rsi_hline_color", "signal_buy_color", "signal_sell_color",
    "ohlc_panel_bg", "ohlc_panel_text", "ohlc_panel_title",
    "ohlc_panel_high", "ohlc_panel_low",
    "badge_positive", "badge_negative", "badge_neutral", "badge_text_color",
    "gauge_bg", "gauge_border", "gauge_tick_color", "gauge_threshold_color",
    "sentiment_line_color", "sentiment_fill_color",
    "font_family",
}
_LIST_KEYS = {"ma_colors", "gauge_steps", "pie_colors"}
_DICT_KEYS = {"boll_colors", "macd_colors", "kdj_colors"}


def test_themes_exist():
    assert isinstance(LIGHT_THEME, dict)
    assert isinstance(DARK_THEME, dict)


def test_identical_key_sets():
    assert set(LIGHT_THEME.keys()) == set(DARK_THEME.keys())


def test_scalar_values():
    for theme in (LIGHT_THEME, DARK_THEME):
        for key in _SCALAR_KEYS:
            val = theme[key]
            assert isinstance(val, str) and len(val) > 0, f"{key} must be non-empty str, got {val!r}"


def test_list_values():
    for theme in (LIGHT_THEME, DARK_THEME):
        for key in _LIST_KEYS:
            val = theme[key]
            assert isinstance(val, list) and len(val) > 0, f"{key} must be non-empty list"
            for item in val:
                assert isinstance(item, str) and len(item) > 0, f"{key} items must be non-empty str"


def test_dict_values():
    for theme in (LIGHT_THEME, DARK_THEME):
        for key in _DICT_KEYS:
            val = theme[key]
            assert isinstance(val, dict) and len(val) > 0, f"{key} must be non-empty dict"
            for sub_key, sub_val in val.items():
                assert isinstance(sub_val, str) and len(sub_val) > 0, (
                    f"{key}.{sub_key} must be non-empty str"
                )


def test_rise_fall_keys():
    assert "rise_color" in LIGHT_THEME
    assert "fall_color" in LIGHT_THEME
    assert "rise_color" in DARK_THEME
    assert "fall_color" in DARK_THEME


def test_get_theme_light():
    assert get_theme("light") is LIGHT_THEME


def test_get_theme_dark():
    assert get_theme("dark") is DARK_THEME


def test_get_theme_default():
    assert get_theme() is LIGHT_THEME


def test_all_keys_covered():
    expected = _SCALAR_KEYS | _LIST_KEYS | _DICT_KEYS
    assert expected == set(LIGHT_THEME.keys()), (
        f"Uncovered keys: {set(LIGHT_THEME.keys()) - expected}"
    )
