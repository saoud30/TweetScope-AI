import csv
import io
import json


TEXT_FIELDS = ("text", "full_text", "tweet_text", "Tweet_Text", "content", "body", "message")
CONTAINER_FIELDS = ("data", "items", "results", "tweets", "posts")
NESTED_OBJECT_FIELDS = ("tweet", "post", "item")


class XquikImportError(ValueError):
    pass


def load_xquik_texts(file_or_bytes):
    payload = _read_text(file_or_bytes).strip()
    if not payload:
        raise XquikImportError("Export is empty.")

    rows = _load_rows(payload)
    texts = [_extract_text(row) for row in rows]
    texts = [text for text in texts if text]
    if not texts:
        raise XquikImportError("Export does not contain tweet text fields.")

    return texts


def _read_text(file_or_bytes):
    if hasattr(file_or_bytes, "getvalue"):
        value = file_or_bytes.getvalue()
    elif hasattr(file_or_bytes, "read"):
        value = file_or_bytes.read()
    else:
        value = file_or_bytes

    if isinstance(value, bytes):
        return value.decode("utf-8-sig")
    if isinstance(value, str):
        return value

    raise XquikImportError("Export must be a text, CSV, JSON, or JSONL file.")


def _load_rows(payload):
    try:
        return _collect_rows(json.loads(payload))
    except json.JSONDecodeError:
        pass

    try:
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except json.JSONDecodeError:
        rows = []
    if rows:
        return _collect_rows(rows)

    return list(csv.DictReader(io.StringIO(payload)))


def _collect_rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for field in CONTAINER_FIELDS:
            nested_value = value.get(field)
            if isinstance(nested_value, list):
                return nested_value
        return [value]

    return []


def _extract_text(row):
    if isinstance(row, str):
        return row.strip()

    if not isinstance(row, dict):
        return ""

    for field in TEXT_FIELDS:
        value = row.get(field)
        if value is not None:
            return str(value).strip()

    for field in NESTED_OBJECT_FIELDS:
        value = row.get(field)
        text = _extract_text(value)
        if text:
            return text

    return ""
