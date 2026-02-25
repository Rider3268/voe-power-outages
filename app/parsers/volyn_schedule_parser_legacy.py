from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from app.config import ParserProfile
from app.core.constants import GPV_GROUPS
from app.core.models import ParsedSchedule
from app.parsers.errors import ParseError


@dataclass(frozen=True)
class GridLines:
    x: list[int]
    y: list[int]


class VolynScheduleParserLegacy:
    """
    Legacy parser reference (single-grid 24-column strategy).

    This class is kept for historical reference and is not used by the app.
    """

    def __init__(self, profile: ParserProfile, timezone_name: str = "Europe/Kyiv") -> None:
        self.profile = profile
        self.timezone_name = timezone_name

    def parse(
        self,
        image_bytes: bytes,
        *,
        region_id: str,
        fetched_at_utc: datetime,
        source_updated_at_utc: datetime | None = None,
    ) -> ParsedSchedule:
        image = self._decode_image(image_bytes)
        table = self._extract_table_roi(image)
        grid = self._detect_grid(table)

        parsed_rows: dict[str, dict[str, str]] = {}
        rows_to_parse = min(len(GPV_GROUPS), self.profile.expected_rows)

        for row_index in range(rows_to_parse):
            group = GPV_GROUPS[row_index]
            parsed_rows[group] = {}
            y1 = grid.y[row_index]
            y2 = grid.y[row_index + 1]
            if y2 <= y1:
                raise ParseError(f"Invalid row bounds detected for {group}: y1={y1} y2={y2}")

            for hour_index in range(self.profile.expected_hours):
                x1 = grid.x[hour_index]
                x2 = grid.x[hour_index + 1]
                if x2 <= x1:
                    raise ParseError(
                        f"Invalid column bounds detected for hour {hour_index + 1}: x1={x1} x2={x2}"
                    )

                cell = table[y1:y2, x1:x2]
                state = self._classify_cell(cell)
                parsed_rows[group][str(hour_index + 1)] = state

        for group in GPV_GROUPS:
            parsed_rows.setdefault(group, {str(hour): "maybe" for hour in range(1, 25)})

        source_dt = source_updated_at_utc or fetched_at_utc
        if source_dt.tzinfo is None:
            source_dt = source_dt.replace(tzinfo=timezone.utc)
        source_dt = source_dt.astimezone(timezone.utc)

        local_tz = ZoneInfo(self.timezone_name)
        local_dt = source_dt.astimezone(local_tz)
        local_midnight = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)

        return ParsedSchedule(
            region_id=region_id,
            last_updated_utc=fetched_at_utc.astimezone(timezone.utc),
            fact_update_text=local_dt.strftime("%d.%m.%Y %H:%M"),
            today_unix=int(local_midnight.timestamp()),
            data=parsed_rows,
        )

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise ParseError("Image is empty")

        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ParseError("Could not decode image")

        return image

    def _extract_table_roi(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            8,
        )

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        best_rect = None
        best_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < best_area:
                continue

            image_area = image.shape[0] * image.shape[1]
            if area < image_area * 0.25:
                continue

            ratio = w / max(h, 1)
            if ratio < 1.1:
                continue

            best_area = area
            best_rect = (x, y, w, h)

        if best_rect is None:
            return image

        x, y, w, h = best_rect
        return image[y : y + h, x : x + w]

    def _detect_grid(self, table: np.ndarray) -> GridLines:
        gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            8,
        )

        h, w = thresh.shape
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, h // 30)))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 30), 1))

        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

        x_positions = self._line_positions(vertical, axis=0)
        y_positions = self._line_positions(horizontal, axis=1)

        x_lines = self._resolve_hour_boundaries(x_positions, width=w)
        y_lines = self._resolve_row_boundaries(y_positions, height=h)

        if len(x_lines) != self.profile.expected_hours + 1:
            raise ParseError(f"Expected {self.profile.expected_hours + 1} x-lines, got {len(x_lines)}")
        if len(y_lines) != self.profile.expected_rows + 1:
            raise ParseError(f"Expected {self.profile.expected_rows + 1} y-lines, got {len(y_lines)}")

        return GridLines(x=x_lines, y=y_lines)

    def _line_positions(self, img: np.ndarray, *, axis: int) -> list[int]:
        projection = img.sum(axis=axis)
        max_value = float(projection.max()) if projection.size else 0.0
        if max_value <= 0:
            return []

        threshold = max_value * 0.35
        active = np.where(projection >= threshold)[0]
        if active.size == 0:
            return []

        groups: list[list[int]] = []
        current = [int(active[0])]
        for pos in active[1:]:
            p = int(pos)
            if p - current[-1] <= 2:
                current.append(p)
            else:
                groups.append(current)
                current = [p]
        groups.append(current)

        return [int(round(float(np.mean(group)))) for group in groups]

    def _resolve_hour_boundaries(self, x_positions: list[int], *, width: int) -> list[int]:
        needed = self.profile.expected_hours + 1
        if len(x_positions) >= needed + 1:
            window = self._best_uniform_window(x_positions, needed)
            if window is not None:
                return window

        left = int(width * 0.20)
        right = width - 1
        return np.linspace(left, right, needed, dtype=int).tolist()

    def _resolve_row_boundaries(self, y_positions: list[int], *, height: int) -> list[int]:
        needed = self.profile.expected_rows + 1
        if len(y_positions) >= needed + 1:
            window = self._best_uniform_window(y_positions, needed)
            if window is not None:
                return window

        top = int(height * 0.12)
        bottom = height - 1
        return np.linspace(top, bottom, needed, dtype=int).tolist()

    def _best_uniform_window(self, positions: list[int], size: int) -> list[int] | None:
        if len(positions) < size:
            return None

        best: list[int] | None = None
        best_score = float("inf")

        for start in range(0, len(positions) - size + 1):
            candidate = positions[start : start + size]
            diffs = np.diff(candidate)
            if diffs.size == 0:
                continue

            mean_diff = float(np.mean(diffs))
            if mean_diff <= 0:
                continue

            std_diff = float(np.std(diffs))
            score = std_diff / mean_diff
            if score < best_score:
                best_score = score
                best = candidate

        return best

    def _classify_cell(self, cell: np.ndarray) -> str:
        if cell.size == 0:
            return "maybe"

        margin_y = max(1, cell.shape[0] // 10)
        margin_x = max(1, cell.shape[1] // 10)
        core = cell[margin_y : cell.shape[0] - margin_y, margin_x : cell.shape[1] - margin_x]
        if core.size == 0:
            core = cell

        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)

        red1 = cv2.inRange(hsv, (0, 70, 40), (12, 255, 255))
        red2 = cv2.inRange(hsv, (160, 70, 40), (180, 255, 255))
        red = cv2.bitwise_or(red1, red2)

        yellow = cv2.inRange(hsv, (15, 60, 80), (45, 255, 255))
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90))

        area = float(core.shape[0] * core.shape[1])
        if area <= 0:
            return "maybe"

        red_ratio = float(np.count_nonzero(red)) / area
        yellow_ratio = float(np.count_nonzero(yellow)) / area
        dark_ratio = float(np.count_nonzero(dark)) / area

        if red_ratio >= self.profile.red_ratio_threshold:
            return "no"

        if yellow_ratio >= self.profile.yellow_ratio_threshold:
            mid_x = core.shape[1] // 2
            left = dark[:, :mid_x]
            right = dark[:, mid_x:]

            left_ratio = float(np.count_nonzero(left)) / max(left.size, 1)
            right_ratio = float(np.count_nonzero(right)) / max(right.size, 1)

            if (
                left_ratio >= self.profile.half_block_ratio_threshold
                and right_ratio < self.profile.half_block_ratio_threshold * 0.7
            ):
                return "first"
            if (
                right_ratio >= self.profile.half_block_ratio_threshold
                and left_ratio < self.profile.half_block_ratio_threshold * 0.7
            ):
                return "second"
            return "maybe"

        blue = cv2.inRange(
            hsv,
            (self.profile.blue_hue_min, 30, 40),
            (self.profile.blue_hue_max, 255, 255),
        )
        blue_ratio = float(np.count_nonzero(blue)) / area
        if blue_ratio >= self.profile.blue_ratio_threshold:
            blue_pixels = hsv[blue > 0]
            if blue_pixels.size > 0:
                sat_mean = float(np.mean(blue_pixels[:, 1]))
                val_mean = float(np.mean(blue_pixels[:, 2]))
                if (
                    sat_mean >= self.profile.blue_off_saturation_threshold
                    and val_mean <= self.profile.blue_off_value_threshold
                ):
                    return "no"
            return "yes"

        if dark_ratio >= self.profile.dark_ratio_threshold and red_ratio > 0.1:
            return "no"

        return "yes"
