from __future__ import annotations

from typing import Final

GPV_GROUPS: Final[list[str]] = [
    "GPV1.1",
    "GPV1.2",
    "GPV2.1",
    "GPV2.2",
    "GPV3.1",
    "GPV3.2",
    "GPV4.1",
    "GPV4.2",
    "GPV5.1",
    "GPV5.2",
    "GPV6.1",
    "GPV6.2",
]

TIME_TYPE: Final[dict[str, str]] = {
    "yes": "Світло є",
    "maybe": "Можливе відключення",
    "no": "Світла немає",
    "first": "Світла не буде перші 30 хв.",
    "second": "Світла не буде другі 30 хв.",
}

TIME_ZONE: Final[dict[str, list[str]]] = {
    str(i): [f"{i - 1:02d}-{i:02d}", f"{i - 1:02d}:00", f"{i:02d}:00" if i < 24 else "24:00"]
    for i in range(1, 25)
}
