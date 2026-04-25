"""Local validation of Gemini structured responses for Stage 4.

Even though the SDK enforces a JSON schema, we revalidate locally to catch
edge cases (truncated outputs, missing fields, unexpected enum values).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.stage4_query.prompt_builder import DIMENSIONS

VALID_SELECTIONS = {"a", "b", "c", "d"}


def validate_response(payload: Any) -> Tuple[bool, str]:
    """Return (is_valid, error_message). Empty error if valid."""
    if not isinstance(payload, dict):
        return False, "Response is not a JSON object"

    activities = payload.get("activities")
    if not isinstance(activities, list):
        return False, "Missing or non-list 'activities' field"

    if len(activities) != 5:
        return False, f"Expected exactly 5 activities, got {len(activities)}"

    seen_numbers: List[int] = []
    for idx, act in enumerate(activities, start=1):
        if not isinstance(act, dict):
            return False, f"Activity #{idx} is not an object"

        number = act.get("activity_number")
        if not isinstance(number, int) or not (1 <= number <= 5):
            return False, f"Activity #{idx} has invalid activity_number: {number!r}"
        seen_numbers.append(number)

        for dim in DIMENSIONS:
            obj = act.get(dim)
            if not isinstance(obj, dict):
                return False, f"Activity #{idx} missing dimension '{dim}'"

            selection = obj.get("selection")
            if selection not in VALID_SELECTIONS:
                return (
                    False,
                    f"Activity #{idx} '{dim}' has invalid selection: {selection!r}",
                )

            description = obj.get("description")
            if not isinstance(description, str) or not description.strip():
                return (
                    False,
                    f"Activity #{idx} '{dim}' has empty/invalid description",
                )

    if sorted(seen_numbers) != [1, 2, 3, 4, 5]:
        return False, f"activity_number values must be 1..5, got {seen_numbers}"

    return True, ""


def normalize_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of `payload` with activities sorted by number."""
    out = dict(payload)
    activities = list(payload.get("activities", []))
    activities.sort(key=lambda a: a.get("activity_number", 0))
    out["activities"] = activities
    return out
