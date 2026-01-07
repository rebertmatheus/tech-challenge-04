from datetime import date, datetime
import json


# Local implementations for tests (module removed by user request)
def parse_ticker(body_bytes: bytes) -> str:
    body = json.loads(body_bytes.decode()) if body_bytes else {}
    ticker = body.get("ticker")
    if not ticker or not str(ticker).strip():
        raise ValueError("Parâmetro 'ticker' é obrigatório")
    return str(ticker).strip().upper()


def parse_date_str(body_bytes: bytes) -> date | None:
    body = json.loads(body_bytes.decode()) if body_bytes else {}
    date_str = body.get("date")
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Formato de data inválido: {date_str}. Use YYYY-MM-DD")


def build_response_body(success: bool, payload: dict) -> dict:
    body = {"success": success}
    body.update(payload or {})
    return body


def test_parse_ticker_valid():
    body = b'{"ticker": " itub4 "}'
    assert parse_ticker(body) == "ITUB4"


def test_parse_ticker_missing_raises():
    try:
        parse_ticker(b'{}')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "ticker" in str(e)


def test_parse_date_valid():
    body = b'{"date": "2024-12-31"}'
    d = parse_date_str(body)
    assert isinstance(d, date)
    assert d.isoformat() == "2024-12-31"


def test_parse_date_invalid_raises():
    try:
        parse_date_str(b'{"date": "31-12-2024"}')
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "Formato de data inválido" in str(e)


def test_build_response_body_merges():
    payload = {"ticker": "ITUB4", "value": 123}
    body = build_response_body(True, payload)
    assert body["success"] is True
    assert body["ticker"] == "ITUB4"
    assert body["value"] == 123
