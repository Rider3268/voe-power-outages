from __future__ import annotations

import numpy as np

from app.config import ParserProfile
from app.parsers.volyn_schedule_parser import VolynScheduleParser


def _profile() -> ParserProfile:
    return ParserProfile(
        expected_rows=12,
        expected_hours=24,
        red_ratio_threshold=0.22,
        yellow_ratio_threshold=0.24,
        half_block_ratio_threshold=0.35,
        dark_ratio_threshold=0.20,
        blue_hue_min=90,
        blue_hue_max=125,
        blue_ratio_threshold=0.40,
        blue_off_saturation_threshold=90.0,
        blue_off_value_threshold=185.0,
    )


def test_classify_sample_power_on_color_as_yes() -> None:
    parser = VolynScheduleParser(_profile())
    # Sample color: RGB(185, 200, 215) from power_on image.
    cell_bgr = np.full((48, 48, 3), (215, 200, 185), dtype=np.uint8)

    state = parser._classify_cell(cell_bgr)

    assert state == "yes"


def test_classify_sample_power_off_color_as_no() -> None:
    parser = VolynScheduleParser(_profile())
    # Sample color: RGB(68, 110, 155) from power_off image.
    cell_bgr = np.full((48, 48, 3), (155, 110, 68), dtype=np.uint8)

    state = parser._classify_cell(cell_bgr)

    assert state == "no"
