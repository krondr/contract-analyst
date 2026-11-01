"""Russian labels shared by the report, consolidation summaries and chat."""

from datetime import date

STATUS_RU = {"active": "действует", "partially_suspended": "часть обязательств прекращена",
             "terminated": "расторгнут", "not_yet_in_force": "ещё не вступил в силу", "unknown": "неизвестно"}
VAT_RU = {"included": "с НДС", "excluded": "без НДС", "unknown": "НДС не указан"}
KIND_RU = {"main": "основной договор", "supplement": "доп. соглашение", "framework": "соглашение", "unknown": "не определён"}
TERM_RU = {"subject": "Предмет", "validity": "Срок действия", "payment_terms": "Порядок расчётов",
           "termination": "Расторжение", "notification_channel": "Уведомления", "penalties": "Штрафы",
           "plan_indicators": "Плановые показатели", "tariff_groups": "Группы тарифов", "trade_points": "Торговые точки",
           "counterparty_type": "Тип контрагента", "city": "Город", "remuneration_model": "Модель вознаграждения",
           "status": "Статус", "year": "Год"}


def status_ru(status: str, since: date | None) -> str:
    """For a contract not in force yet the date is when it starts: "(с <date>)" would read as "not in force since"."""
    label = STATUS_RU.get(status, status)
    if since is None:
        return label
    return f"{label}; вступает в силу {since}" if status == "not_yet_in_force" else f"{label} с {since}"
