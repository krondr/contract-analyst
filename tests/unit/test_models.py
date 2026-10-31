from datetime import date
from decimal import Decimal

from contract_analyst.models import Extracted, RateRow, ToolResult, ValueStatus


def test_rate_row_roundtrip_keeps_decimal_and_date():
    row = RateRow(label="Тариф A", amount=Decimal("330.5"), vat="excluded",
                  valid_from=date(2026, 7, 1), status=ValueStatus.ACCEPTED)
    data = row.model_dump(mode="json")
    assert data["amount"] == "330.5" and data["valid_from"] == "2026-07-01"
    assert RateRow.model_validate(data) == row


def test_generic_extracted_coerces_type():
    assert Extracted[date](value="2021-01-21", status="accepted").value == date(2021, 1, 21)


def test_tool_result_has_contract_fields():
    r = ToolResult(status="error", error_type="no_text_layer", hint="попробуй rerecognize")
    assert set(r.model_dump()) == {"status", "data", "error_type", "hint"}
