"""Realtime tools (read-only).

The project has **no** weather or visitor-flow integration yet. These tools therefore
read the ``realtime_status`` table and, when it holds nothing for the requested
subject, answer ``available=false`` with an explicit "no realtime data" message.
Guessing weather or crowd levels is exactly what a scenic-area assistant must not do.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import text

from ..core.config import PARK_ID
from ..core.db import engine, payload_rows
from .registry import tool

#: Published service-centre phone number, also used by the knowledge base.
SERVICE_PHONE = "0837-7739753"

#: Metric families the table may hold, mapped to an external data source we do not have.
EXTERNAL_METRICS = {
    "weather": "天气",
    "visitor_flow": "客流",
    "ticket_inventory": "余票",
    "road_condition": "道路通行",
}


class GetRealtimeStatusArgs(BaseModel):
    subject: str | None = Field(
        default=None,
        max_length=60,
        description="要查询的对象或指标关键词，例如「天气」「客流」「五花海」「观光车」；留空返回全部已知状态",
    )
    include_internal_status: bool = Field(
        default=True, description="是否附带知识库中记录的景点与设施开放状态"
    )


class GetParkNoticesArgs(BaseModel):
    keyword: str | None = Field(default=None, max_length=60, description="公告关键词，例如「轮休」「关闭」")
    limit: int = Field(default=10, ge=1, le=50)


def _rows(statement: str, params: dict[str, Any]) -> list[Any]:
    """Rows for one realtime query.

    The expected unavailability case is already handled by the explicit guard, so a
    failure here is a bug (a renamed column, a driver error) rather than an outage. It is
    allowed to propagate so the tool registry turns it into a visible ``error`` outcome;
    swallowing it made the tool tell the visitor "this park has no realtime data source"
    when the truth was that our own query was broken.
    """
    from ..core.db import enabled

    if not enabled():
        return []
    with engine.connect() as conn:
        return conn.execute(text(statement), params).fetchall()


def _internal_status() -> dict[str, Any]:
    attractions = payload_rows("attractions", 100)
    facilities = payload_rows("facilities", 100)
    closed = [item.get("name") for item in attractions if item.get("status") != "open"]
    seasonal = [
        {"name": item.get("name"), "seasonal_closure": item.get("seasonal_closure")}
        for item in attractions
        if item.get("seasonal_closure")
    ]
    unavailable = [item.get("name") for item in facilities if item.get("status") != "available"]
    return {
        "closed_attractions": closed,
        "seasonal_closure": seasonal,
        "unavailable_facilities": unavailable,
        "attractions_checked": len(attractions),
        "facilities_checked": len(facilities),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@tool(
    name="get_realtime_status",
    description=(
        "查询实时状态（天气、客流、余票、道路通行）以及知识库中记录的景点/设施当前开放状态。"
        "未接入外部数据源时返回 available=false，此时必须如实告知游客「暂无实时数据」。"
    ),
    args_schema=GetRealtimeStatusArgs,
    allowed=("realtime_agent", "route_agent"),
    tags=("read", "realtime"),
)
def get_realtime_status(args: GetRealtimeStatusArgs) -> dict[str, Any]:
    rows = _rows(
        """SELECT subject_type, subject_id, metric, value, source, observed_at, expires_at
           FROM realtime_status WHERE park_id=:park_id ORDER BY observed_at DESC LIMIT 50""",
        {"park_id": PARK_ID},
    )
    records = [
        {
            "subject_type": row[0],
            "subject_id": row[1],
            "metric": row[2],
            "value": row[3],
            "source": row[4],
            "observed_at": str(row[5]),
            "expires_at": str(row[6]) if row[6] else None,
        }
        for row in rows
    ]
    now = datetime.now(timezone.utc)
    for record in records:
        expires_at = record.get("expires_at")
        try:
            parsed = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00")) if expires_at else None
            record["stale"] = bool(parsed and parsed <= now)
        except (TypeError, ValueError):
            record["stale"] = False
    subject = (args.subject or "").strip()
    if subject:
        records = [
            record
            for record in records
            if subject in str(record.get("metric") or "")
            or subject in str(record.get("subject_id") or "")
            or subject in str(record.get("value"))
        ]

    active_records = [record for record in records if not record.get("stale")]
    payload: dict[str, Any] = {
        "requested_subject": subject or None,
        "available": bool(active_records),
        "records": records,
        "external_metrics": EXTERNAL_METRICS,
        "consult_phone": SERVICE_PHONE,
    }
    if not active_records:
        payload["summary"] = (
            f"九寨沟景区暂无未过期的「{subject or '实时状态'}」数据，"
            f"请以景区现场公告为准（咨询电话 {SERVICE_PHONE}）。"
        )
    else:
        payload["summary"] = "；".join(
            f"{record['metric']}={record['value']}（来源 {record['source']}，观察于 {record['observed_at']}）"
            for record in active_records[:3]
        )
    if args.include_internal_status:
        payload["internal_status"] = _internal_status()
        payload["internal_status_note"] = "以上为知识库记录的状态，非实时数据。"
    return payload


@tool(
    name="get_park_notices",
    description="查询景区公告与临时关闭/轮休保育信息（来自 realtime_status 的 notice 记录）。",
    args_schema=GetParkNoticesArgs,
    allowed=("realtime_agent",),
    tags=("read", "realtime"),
)
def get_park_notices(args: GetParkNoticesArgs) -> dict[str, Any]:
    rows = _rows(
        """SELECT subject_id, metric, value, source, observed_at FROM realtime_status
           WHERE park_id=:park_id AND metric LIKE 'notice%' ORDER BY observed_at DESC LIMIT :limit""",
        {"park_id": PARK_ID, "limit": args.limit},
    )
    notices = [
        {"subject_id": row[0], "metric": row[1], "value": row[2], "source": row[3], "observed_at": str(row[4])}
        for row in rows
    ]
    if args.keyword:
        notices = [notice for notice in notices if args.keyword in str(notice)]
    return {
        "available": bool(notices),
        "total": len(notices),
        "notices": notices,
        "summary": (
            "；".join(str(notice["value"]) for notice in notices[:3])
            if notices
            else f"公告数据源尚未接入，无法确认临时关闭信息，请以景区现场公告为准（咨询电话 {SERVICE_PHONE}）。"
        ),
    }
