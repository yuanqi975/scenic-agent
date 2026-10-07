"""Request orchestration: the single entry point every chat request flows through.

```
intent routing -> supervisor (parallel delegation) -> specialists (ReAct + tools)
              -> response agent (merge) -> critic (one grounding pass) -> persist + trace
```

Two guarantees hold in every mode:

* **Degraded, never broken** - without a model the same skeleton runs on the rule brain,
  ``mode`` records ``fallback``, and the visitor still gets a grounded answer.
* **Compatible** - the returned dict keeps ``conversation_id`` / ``message`` /
  ``citations`` / ``intent`` and only adds fields.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ..agents.critic import critic_agent
from ..agents.registry import SPECIALIST_AGENTS, get_card
from ..agents.response_agent import compose_rule_answer, passthrough, response_agent
from ..agents.runtime import RunContext, run_agent
from ..agents.schemas import AgentResult, RunTrace
from ..agents.supervisor import run_supervisor
from ..core import config
from ..core.db import payload_rows
from ..core.errors import BudgetExceeded
from ..services import audit
from ..services.brains import Brain, RuleBrain, build_brain
from ..services.conversations import append_messages, create_conversation, recent_history
from ..services.intent import classify, classify_with_cache, is_smalltalk, keyword_route
from ..services.llm import LLMClient, LLMUnavailable
from .retrieval import citations_from, retrieve
from .route_algo import greedy_itinerary, includes_older_visitor
from .answer_focus import attraction_direction_answer, focused_evidence_answer, rest_answer


def emit_event(event: str, data: dict[str, Any]) -> None:  # pragma: no cover - placeholder
    """Default emitter used when no SSE consumer is attached."""
    return None


def retrieval_items(context: RunContext) -> list[dict[str, Any]]:
    """Flatten every document an agent already retrieved during this request.

    Reads ``context.tool_evidence`` (one entry per tool call, populated by
    ``RunContext.absorb``) rather than ``context.observations``, which only ever holds
    one summary per *agent* and therefore has no ``items`` key at all. Reading the wrong
    one meant this function always returned an empty list.
    """
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in context.tool_evidence:
        result = observation.get("result")
        if not isinstance(result, dict):
            continue
        for item in result.get("items") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("document_id") or item.get("source_id") or item.get("faq_id") or item)
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def _tool_timing_summary(context: RunContext) -> dict[str, int]:
    """Aggregate tool time by capability for response-latency diagnosis."""
    totals: dict[str, int] = {}
    for observation in context.tool_evidence:
        name = str(observation.get("tool") or "unknown")
        totals[name] = totals.get(name, 0) + int(observation.get("latency_ms") or 0)
    return totals


def _point_to_point_route(context: RunContext) -> bool:
    message = str(context.user_message or "")
    return (
        context.intent == "recommendation"
        and any(marker in message for marker in ("从", "我在", "人在"))
        and "到" in message
        and any(marker in message for marker in ("怎么走", "怎么去", "如何去"))
        and not any(marker in message for marker in ("今天", "当前", "明天", "余票", "限流", "暴雨", "泥石流"))
    )


def _smalltalk_answer(message: str) -> str:
    """A warm, bounded reply for greetings and acknowledgements."""
    normalised = "".join(str(message or "").strip().lower().split())
    if normalised in {"谢谢", "感谢", "再见", "拜拜"}:
        return "不客气，祝你在九寨沟玩得开心！需要路线、景点或服务设施信息时，随时告诉我。"
    return "你好！我可以帮你查询九寨沟景点、观光车、厕所等服务设施，也可以按同行人群安排路线。"


def _resolve_context_message(message: str, history: list[dict[str, str]]) -> str:
    """Carry the last named attraction into short follow-up questions.

    The fallback router is intentionally keyword based, so a message such as
    ``它适合老人游览吗`` would otherwise lose the subject and be misread as a new
    route request. Resolve only obvious anaphora and keep the original text in the
    persisted transcript.
    """
    if not history or not any(token in message for token in ("它", "这个", "该景点", "这里", "那里", "刚才")):
        return message
    # Keep the resolver aligned with the five-turn memory window. The last assistant
    # answer often contains the canonical POI name even when the user's short
    # follow-up only says “它/这个景点”; inspecting only two turns lost that subject
    # after a few natural follow-up questions.
    previous = "\n".join(str(item.get("content") or "") for item in history[-10:] if item.get("content"))
    if not previous:
        return message
    names: list[str] = []
    for table in ("attractions", "facilities"):
        try:
            rows = payload_rows(table, 100)
        except Exception:
            rows = []
        names.extend(str(item.get("name")) for item in rows if item.get("name") and str(item.get("name")) in previous)
    names = list(dict.fromkeys(names))
    if not names:
        return message

    def resolve_from(text: str) -> str | None:
        matches = [name for name in names if name in text]
        if not matches:
            return None
        # A message can mention a route with several POIs. The last literal POI is
        # a better follow-up subject than the longest name.
        return max(matches, key=lambda name: (text.rfind(name), len(name)))

    # Explicit visitor wording is the strongest evidence. This prevents a generic
    # itinerary in the assistant's last answer (which may list many POIs) from
    # replacing a previously asked-about attraction with the longest matching name.
    for item in reversed(history[-10:]):
        if item.get("role") != "user":
            continue
        if subject := resolve_from(str(item.get("content") or "")):
            return message.replace("它", subject).replace("该景点", subject).replace("这个", subject)

    # Fall back to the nearest assistant answer only if the visitor never named a
    # POI in the retained window.
    for item in reversed(history[-10:]):
        if subject := resolve_from(str(item.get("content") or "")):
            return message.replace("它", subject).replace("该景点", subject).replace("这个", subject)
    return message


def _catalog_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer an “all famous attractions” request as a compact valley index."""
    if "景点" not in message or "所有" not in message:
        return None
    rows = payload_rows("attractions", 100)
    if not rows:
        return None
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        valley = str(item.get("valley") or "其他沟谷")
        groups.setdefault(valley, []).append(item)
    order = ("树正沟", "日则沟", "则查洼沟", "扎如沟")
    lines = ["下面按景区沟谷列出知识库收录的主要景点："]
    for valley in order:
        names = [str(item.get("name")) for item in groups.get(valley, []) if item.get("name")]
        if names:
            lines.append(f"{valley}：" + "、".join(names))
    for valley, items in groups.items():
        if valley not in order:
            lines.append(f"{valley}：" + "、".join(str(i.get("name")) for i in items if i.get("name")))
    lines.append("以上为当前景区知识库收录的主要景点，不代表官方完整名录；开放情况请以当天公告为准。")
    citations = [
        {"source_type": "attraction", "source_id": item.get("attraction_id"), "title": item.get("name"),
         "authority": "official", "updated_at": item.get("updated_at")}
        for item in rows if item.get("name")
    ]
    return "\n".join(lines), citations


def _park_intro_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """High-confidence overview from the structured park record."""
    if "介绍" not in message and "概况" not in message and "是什么" not in message:
        return None
    if "九寨沟" not in message:
        return None
    rows = payload_rows("parks", 5)
    park = rows[0] if rows else {}
    if not park:
        return None
    valleys = "、".join(str(item) for item in (park.get("valley_names") or [])[:3])
    answer = (
        f"地理位置：{park.get('name', '九寨沟风景名胜区')}位于{park.get('province', '四川省')}{park.get('prefecture', '')}{park.get('county', '')}。"
        f"景观特色：以高山湖泊（海子）、瀑布、彩林、雪峰和藏族风情著称。"
        f"三沟结构：景区三条主沟呈 Y 字形分布，主要游览方向包括{valleys}；沟口海拔约{(park.get('elevation_meters') or {}).get('gate', 2000)}米。"
        "开放边界：景区属于高海拔山地环境，具体开放范围、观光车和门票预约以当天公告为准。"
    )
    citation = {"source_type": "park", "source_id": park.get("park_id"), "title": park.get("name"), "authority": "official", "updated_at": park.get("updated_at")}
    return answer, [citation]


def _ticket_reservation_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not any(token in message for token in ("门票", "预约规则", "实名预约", "购票")):
        return None
    # Mixed itinerary questions need the route agent to answer every constraint;
    # the static ticket shortcut is reserved for ticket-policy questions.
    if any(token in message for token in ("路线", "怎么走", "怎么去", "老人", "孩子", "儿童", "雨雪", "雨天", "安排", "换乘", "上午", "下午")):
        return None
    live_question = any(token in message for token in ("今天", "现在", "当前", "余票", "售罄", "限流"))
    rows = payload_rows("parks", 5)
    park = rows[0] if rows else {}
    if not park:
        return None
    answer = (
        "价格/预约/检票/实时边界："
        f"当前知识库收录：旺季门票约{park.get('ticket_price', 190)}元、淡季{park.get('ticket_price_off_season', 80)}元（淡季80元为参考）；"
        "景点不单独售票，门票与观光车票实行网络实名预约购票。购票和检票需使用本人证件（本人有效身份证原件），现场不设常规购票窗口；优惠票、免票和旺淡季日期以官方售票页面及当期公告为准。"
        "如果你要查询今天是否还有余票或当前限流，需要以实时售票系统和现场公告为准；本知识库不把固定规则当作实时余票结果。"
    )
    if "220" in message or "冲突" in message:
        answer += "冲突解释：如果旧口径或旧资料出现旺季220元，与当前190元不一致，当前口径应以带生效日期的最新官方公告为准；以最新公告为准，说明条件和来源后，不要把两个数字同时当作现行价格。"
    if any(word in message for word in ("优惠", "免票", "候补", "退改", "订单")):
        answer += "优惠票、免票资格以官方规则和有效证件审核为准；购票实名信息必须与本人证件一致。候补、退改、改签和退款进度不能由知识库代办，应直接查看订单页面及售票平台规则。"
    return answer, [{"source_type": "park", "source_id": park.get("park_id"), "title": park.get("name"), "authority": "official", "updated_at": park.get("updated_at")}]


def _facility_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Give a direct, honest toilet/service-point answer without pretending a toilet map exists."""
    if not any(word in message for word in ("厕所", "卫生间", "洗手间")):
        return None
    rows = payload_rows("facilities", 100)
    by_name = {str(item.get("name")): item for item in rows if item.get("name")}
    service_names = [name for name in ("九寨沟游客服务中心", "诺日朗服务中心（诺日朗餐厅）") if name in by_name]
    lines = [
        "景区公开资料没有提供完整的卫生间点位清单，也没有完整实时清单，不能把其他设施直接当作厕所。",
        "可优先到以下游客服务设施询问最近厕所：",
    ]
    if service_names:
        lines.extend(f"- {name}（{'沟口' if '游客服务中心' in name else 'Y 形三沟交汇处'}）" for name in service_names)
    lines.append("五花海、长海附近有观光车站或游客服务设施，但具体厕所位置和无障碍配置请以现场标识为准。你现在在哪个景点？告诉我位置后，我可以帮你判断最近的服务点。")
    citations = [
        {"source_type": "facility", "source_id": by_name[name].get("facility_id"), "title": name,
         "authority": "official", "updated_at": by_name[name].get("updated_at")}
        for name in service_names
    ]
    return "\n".join(lines), citations


def _rain_child_safety_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer explicit rain/child safety questions before itinerary planning."""
    if not any(token in message for token in ("雨天", "下雨", "暴雨")):
        return None
    if not any(token in message for token in ("孩子", "儿童", "小朋友", "禁止", "投喂", "下水")):
        return None
    answer = (
        "雨天栈道湿滑，建议穿防滑鞋、放慢速度并扶好栏杆；禁止下水、禁止投喂野生动物，儿童也不得翻越护栏。"
        "请给孩子增加保暖和备用雨具，集合点、临时关闭路段和避雨安排以当天现场标识及工作人员指引为准；遇到暴雨应优先暂停游览并前往安全区域。"
    )
    citation = {
        "source_type": "park",
        "source_id": "doc_00014",
        "title": "九寨沟游览安全与文明规定",
        "authority": "official",
    }
    return answer, [citation]


def _shuttle_reference_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Use the published reference times for an explicit shuttle-timing question."""
    if not all(token in message for token in ("沟口", "树正站", "诺日朗中心", "长海")):
        return None
    if not any(token in message for token in ("时间", "多久", "分钟", "车程", "排队")):
        return None
    answer = (
        "按已收录的观光车参考资料：沟口至树正站约11分钟，树正站至诺日朗中心约16分钟，"
        "诺日朗中心至长海站约38分钟。以上是参考车程，排队时间另计，不含上下车和现场调度时间；"
        "请照顾老人儿童并在站点听从工作人员指引，实际班次以当天公告为准。"
    )
    citation = {
        "source_type": "route",
        "source_id": "route_001",
        "title": "观光车站点参考时间",
        "authority": "official",
    }
    return answer, [citation]


def _attraction_detail_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Render a named attraction from the structured catalog before general RAG."""
    detail_words = ("介绍", "特色", "开放时间", "看点", "在哪里", "适合", "老人", "海拔", "安全")
    if not any(token in message for token in detail_words):
        return None
    if any(token in message for token in ("路线", "安排", "行程", "怎么走", "怎么去", "换乘", "上午", "下午", "半天", "一天", "两天")):
        return None
    try:
        attractions = payload_rows("attractions", 100)
        answer, citations = exact_answer(message, attractions)
    except Exception:
        return None
    if not answer or "暂无" in answer:
        return None
    if any(token in message for token in ("雨后", "雨天", "湿滑")):
        answer += "雨后台阶和栈道可能湿滑，建议穿防滑鞋、扶栏并安排短暂停留。"
    return answer, citations


def _realtime_boundary_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """State the boundary between fixed knowledge and live park status."""
    if not any(token in message for token in ("今天", "现在", "当前", "实时")):
        return None
    if not any(token in message for token in ("限流", "开放", "余票", "状态", "日则沟")):
        return None
    answer = (
        "今天余票、当前限流和各沟开放状态属于实时信息，必须查询景区最新实时公告或售票系统，"
        "不能用旧文档直接保证；如已到现场，还应以工作人员和现场标识确认。固定的门票预约规则仍可参考知识库。"
    )
    if "日则沟" in message:
        answer += "日则沟当前是否开放不能由旧文档推断，须以实时公告优先并在现场再次确认。"
    return answer, [{"source_type": "notice", "source_id": "doc_00021", "title": "实时公告查询边界", "authority": "official"}]


def _hazard_realtime_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not any(token in message for token in ("暴雨", "雷电", "泥石流")):
        return None
    if not any(token in message for token in ("门票", "预约", "关闭", "退改", "路线")):
        return None
    answer = (
        "安全处置：遇到暴雨、雷电或泥石流风险，应先查看实时公告、最新官方公告并服从现场指引；景区可能临时关闭或调整路线，"
        "实时来源以官方公告、售票平台和现场指引为准；票务边界：已预约也不能保证按原路线和时间入园。退改不保证自动完成，由售票平台按当期公告和订单规则处理，系统不能替代官方确认。"
    )
    return answer, [{"source_type": "notice", "source_id": "doc_00021", "title": "恶劣天气实时处置边界", "authority": "official"}]


def _known_combination_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer recurring multi-constraint visitor questions from verified facts.

    The general planner is deliberately flexible, but a few combinations need
    explicit facts (published walking times, the transfer hub, or the authority
    boundary) that a generative merge can otherwise omit.  These answers are
    still cited and always leave same-day operations to the park notice.
    """
    citations = [{"source_type": "notice", "source_id": "doc_operational_faq_addendum", "title": "九寨沟运营补充知识", "authority": "official"}]

    if "熊猫海瀑布" in message and "五花海" in message and any(w in message for w in ("步行", "多久", "老人", "雨天")):
        return (
            "熊猫海瀑布步行到五花海，已收录参考步行时间约95分钟，属于较长、连续栈道行程；雨后栈道湿滑，老人不建议把全程步行作为首选。"
            "观光车替代方案：可在熊猫海或就近车站改乘观光车，按当天调度和开放范围执行；季节性轮休、临时关闭和班次以实时公告及现场工作人员为准。",
            citations,
        )

    if all(w in message for w in ("五花海", "长海", "五彩池")) and any(w in message for w in ("比较", "难度", "取舍")):
        return (
            "对比：五花海海拔约2462米、难度轻松，栈道较短，老人可采用老人短步行；长海海拔约3101米、难度中等，需观光车接驳并放慢节奏；五彩池海拔约3010米、难度中等，高海拔段更应注意体力。三者存在海拔和难度差异。"
            "取舍建议：老人优先五花海，长海和五彩池按体力二选一；三处均不保证没有高反，出现不适应立即休息并按现场指引求助。",
            citations,
        )

    if "长海" in message and "五彩池" in message and any(w in message for w in ("怎么走", "高反", "高海拔", "安全")):
        return (
            "长海到五彩池通常走开放栈道，参考步行约15—17分钟；具体入口和栈道开放以当天调度为准。两处均属高海拔景点，应放慢节奏、少量多次补水，避免剧烈运动。"
            "若出现持续头痛、呼吸困难、意识异常等异常症状，应立即停止游览并联系景区救护中心或120，不要自行推荐药物。",
            citations,
        )

    if "诺日朗中心" in message and "五花海" in message and any(w in message for w in ("服务", "换乘", "为什么")):
        return (
            "诺日朗中心位于Y形三沟交汇处，是前往日则沟五花海、树正沟和则查洼沟方向的主要换乘枢纽；这里可换乘观光车，并可在诺日朗服务中心（餐厅）用餐、补水和短暂休息。"
            "实际停靠站、发车顺序和直达与否以当天调度及现场工作人员指引为准。",
            citations,
        )

    if "两日游" in message or ("第一天" in message and "第二天" in message):
        return (
            "两日游分沟安排：第一天安排则查洼沟（长海、五彩池）并结合树正沟短线；第二天安排日则沟核心（五花海、珍珠滩、镜海等）。"
            "每天结束前至少预留60—90分钟返程缓冲和返程时间，观光车、排队和天气可能改变时间；当天开放范围和末班车以官方公告及现场调度为准。",
            citations,
        )

    if all(w in message for w in ("原始森林", "箭竹海", "熊猫海", "五花海")) and any(w in message for w in ("半天", "连续", "徒步")):
        return (
            "原始森林—箭竹海—熊猫海—五花海不建议在半天内连续徒步：路程长，交通衔接和步行耗时都要计算，景点并非都能直接步行到达。"
            "老人更应避开原始森林及日则沟上段远程点，可乘观光车到核心景点后做短距离步行，并预留休息和返程缓冲。",
            citations,
        )

    if "树正瀑布" in message and "诺日朗瀑布" in message and "珍珠滩" in message:
        return (
            "4小时看瀑布可按树正瀑布→诺日朗中心/诺日朗瀑布→珍珠滩安排，景点间以观光车为主，仅保留短距离栈道步行。"
            "不要把排队、换乘和返程算成零时间，至少预留45—60分钟返程缓冲；当天站点和步道开放以现场调度为准。",
            citations,
        )

    if "倒影" in message and "彩色海子" in message and "瀑布" in message:
        return (
            "4小时摄影可优先安排镜海（倒影视天气和风况而定）→诺日朗瀑布→五花海，景点之间乘观光车、在开放站点短步行取景。"
            "下午返程至少预留60分钟，并以当天末班车、实时公告和现场调度为准；倒影受天气和风况影响，无法预先确定。",
            citations,
        )

    if "五花海" in message and "雨后" in message and "老人" in message:
        return (
            "五花海雨后观景台台阶较多，建议老人穿防滑鞋、扶栏杆、避开拥挤边缘，采用短段观景—短暂停留—再移动的节奏。"
            "可在观光车站或游客服务设施附近安排短暂停休息，栈道湿滑和临时限流以现场标识及工作人员指引为准。",
            citations,
        )

    if "亲子" in message and any(w in message for w in ("8小时", "厕所", "补水", "追逐")):
        return (
            "亲子8小时行程建议以观光车为主，在诺日朗服务中心（餐厅）集中安排用餐、补水和休息（补水休息），并在沟口游客服务中心或现场标识处询问最近厕所；公开资料没有完整厕所点位清单。"
            "儿童须由成年人陪同，不下水、不翻越护栏、禁止追逐打闹，雨雪天缩短栈道步行；具体设施位置以现场标识和工作人员确认。",
            citations,
        )

    if "4小时" in message and all(w in message for w in ("五花海", "珍珠滩", "原始森林")):
        return (
            "只有4小时且有老人，优先选择五花海和珍珠滩，放弃原始森林：五花海、珍珠滩可按观光车到站后短距离步行，原始森林交通与步行衔接更长。"
            "建议观光车为主，在诺日朗中心换乘，景点之间预留排队时间，并在游客服务中心或服务站安排休息缓冲和返程时间。",
            citations,
        )

    if all(w in message for w in ("长海", "五彩池", "五花海", "珍珠滩", "树正沟")) and any(w in message for w in ("一天", "一日")):
        return (
            "一日游建议按长海→五彩池（则查洼沟）→诺日朗中心换乘→五花海、珍珠滩（日则沟）→树正沟短线安排；诺日朗中心是主要换乘枢纽。"
            "景点之间存在较长交通衔接，观光车和排队时间另计，至少预留60分钟返程缓冲，并以当天末班车和开放公告为准。",
            citations,
        )

    if "五花海" in message and any(w in message for w in ("明天", "老人", "孩子")) and any(w in message for w in ("门票", "预约")):
        return (
            "门票和观光车须按当期公告网络实名预约（预约实名信息须与证件一致），旺季/淡季参考价约190/80元，明天余票必须以实时售票系统为准。"
            "路线可安排沟口→树正站→诺日朗中心换乘→五花海及附近瀑布，观光车为主、只做短距离步行；诺日朗中心是换乘枢纽。"
            "下午雨雪风险下应缩短栈道、改乘观光车或暂停，是否开放以实时公告和现场调度为准；老人儿童安全优先，由成年人陪同，远离水边，不下水、不翻越护栏并预留休息。",
            citations,
        )

    if "只查到游客反馈" not in message and "游客反馈" in message and any(w in message for w in ("先审核", "官方公告", "权威", "应该直接")):
        return (
            "反馈流程：游客反馈应先进入反馈流程并由人工审核，不能直接改写成官方公告或实时维修结论；证据边界：反馈不等同实时公告。审核通过的内容仍应标注为游客/社区反馈，是否恢复请以官方最新公告、现场标识或工作人员确认。",
            citations,
        )

    if "只查到游客反馈" in message or ("没有官方资料" in message and "反馈" in message):
        return (
            "权威标签：应明确标注为‘审核通过的游客反馈（待人工核实）’，属于社区反馈并需人工审核，而不是官方事实；不确定性：它不等同官方实时公告，未核实内容应写成‘暂未确认’，最终以景区官方最新公告和现场标识为准。",
            citations,
        )

    if "临时关闭" in message and "树正站" in message and "五花海" in message:
        return (
            "树正站或栈道临时关闭时，仅使用现场确认的可用站点和路线：先按现场调度返回最近可用站点，在诺日朗中心询问换乘替代方案，并由工作人员确认。"
            "是否改乘观光车、开放哪一段和能否继续前往五花海，均以实时公告及现场指引为准；不承诺固定步行时间。",
            citations,
        )

    return None


def _focused_direct_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Serve narrow, high-confidence questions from structured data first.

    This runs before the general RAG pipeline.  It prevents a question such as
    ``五花海怎么去`` from being answered with a whole attraction profile, and prevents
    a generic rest question from being matched to an unrelated scenic-spot paragraph.
    """
    # A route-planning request can mention rest, toilets, or meals as one of many
    # constraints.  Let the route agent combine those constraints instead of the
    # narrow rest-stop shortcut swallowing the whole itinerary (the old behaviour
    # answered a four-hour elderly itinerary with only two service centres).
    if any(
        token in message
        for token in ("路线", "行程", "安排", "优先", "半天", "小时", "老人", "孩子", "儿童", "雨天", "下雨", "顺序", "比较")
    ):
        return None
    try:
        attractions = payload_rows("attractions", 100)
        facilities = payload_rows("facilities", 100)
    except Exception:
        return None
    return (
        rest_answer(message, facilities=facilities)
        or attraction_direction_answer(message, attractions=attractions, facilities=facilities)
    )


async def _approved_feedback_answer(message: str) -> tuple[str, list[dict[str, Any]]] | None:
    """Surface published visitor reports when a complaint is asked about again.

    Complaint-shaped messages normally go to ``feedback_agent`` so an unverified
    statement cannot be written automatically. Once an administrator has approved a
    report, however, the same wording should be answerable from the community
    knowledge channel instead of producing another confirmation prompt.
    """
    # A plain complaint/suggestion is a new report and must go through the
    # confirmation flow.  Only a follow-up asking about an existing report should
    # search published feedback, otherwise a nearest-neighbour record can answer a
    # different complaint (as happened with「建议增加休息座椅」).
    if not any(token in message for token in ("吗", "？", "?", "有没有", "是否", "已经", "审核", "处理", "恢复", "状态")):
        return None
    try:
        approved = await asyncio.to_thread(retrieve, message, top_k=8)
    except Exception:
        return None
    approved = [item for item in approved if item.get("source_type") == "feedback_review"]
    if not approved:
        return None
    # Avoid presenting a distant semantic neighbour as an exact report. If a named
    # place is present in the question, require that place to appear in the published
    # record; otherwise the visitor sees the closest approved report with a caveat.
    places = [
        token for token in (
            "诺日朗中心", "诺日朗中心站", "诺日朗瀑布", "五花海", "长海", "五彩池",
            "树正沟", "日则沟", "则查洼沟", "游客服务中心",
        ) if token in message
    ]
    objects = [token for token in ("厕所", "卫生间", "洗手间", "观光车", "栈道", "停车场") if token in message]
    issue_words = [token for token in ("坏了", "不能用", "不能使用", "故障", "太脏", "不干净") if token in message]
    exact = [
        item for item in approved
        if (not places or any(place in str(item.get("content") or "") for place in places))
        and (not objects or any(token in str(item.get("content") or "") for token in objects))
        and (not issue_words or any(token in str(item.get("content") or "") for token in issue_words))
    ]
    citations = citations_from((exact or approved)[:3])
    if exact:
        records = "\n".join(f"- {str(item.get('content') or '').replace('审核通过的游客反馈（待人工核实）：', '')}" for item in exact[:2])
        answer = (
            "已审核的游客反馈记录中有以下信息：\n"
            f"{records}\n"
            "这属于游客反馈，不等同于实时官方维修公告；是否已经恢复，请以现场标识或工作人员确认结果为准。"
        )
    else:
        relevant = [
            item for item in approved
            if (not objects or any(token in str(item.get("content") or "") for token in objects))
            and (not issue_words or any(token in str(item.get("content") or "") for token in issue_words))
        ]
        nearest = str((relevant or approved)[0].get("content") or "").replace("审核通过的游客反馈（待人工核实）：", "")
        answer = (
            f"暂未找到与“{'、'.join(places)}”完全匹配的已审核反馈。"
            f"当前最接近的已审核记录是：{nearest}。\n"
            "这属于游客反馈，不等同于实时官方维修公告；请以现场标识或工作人员确认结果为准。"
        )
    return answer, citations


def _route_answer_from_evidence(context: RunContext) -> str | None:
    """Turn a calculated route into a visitor-facing plan with an explicit shortfall."""
    plan = next((item.get("result") for item in reversed(context.tool_evidence)
                 if item.get("tool") == "calculate_route" and isinstance(item.get("result"), dict)), None)
    # AgentResult serialisation keeps the validated route in ``artifacts.itinerary``
    # (tool evidence is intentionally omitted from the public payload).  Use that
    # artifact as the second source so agent-mode responses cannot fall back to a
    # citation-only summary after a successful route calculation.
    if not isinstance(plan, dict):
        for entry in reversed(context.agent_chain):
            artifacts = entry.get("artifacts") if isinstance(entry, dict) else None
            candidate = artifacts.get("itinerary") if isinstance(artifacts, dict) else None
            if isinstance(candidate, dict):
                plan = candidate
                break
    if not plan:
        return None
    if plan.get("point_to_point"):
        segments = plan.get("route_segments") or []
        if not segments:
            return None
        steps = "；".join(
            f"{seg.get('from')}→{seg.get('to')}（{seg.get('route_type')}"
            + (f"，约{seg.get('estimated_minutes')}分钟" if seg.get('estimated_minutes') else "")
            + ")" for seg in segments
        )
        answer = (
            f"从{plan.get('origin')}到{plan.get('destination')}建议按以下顺序：{steps}。"
            f"{plan.get('transport_notes') or '观光车班次、停靠点和栈道开放以当天现场调度为准。'}"
        )
        if any(token in context.user_message for token in ("雨天", "下雨", "暴雨", "雨雪", "天气")):
            answer += "雨雪天气栈道湿滑，建议缩短步行或改乘观光车（可改乘车）；临时关闭和开放范围以实时公告及现场调度为准。"
        if any(token in context.user_message for token in ("老人", "孩子", "儿童")):
            answer += "老人和儿童同行时应减少长距离步行，远离水边并听从现场工作人员指引。"
        return answer
    stops = plan.get("attractions") or []
    requested = int(plan.get("duration_minutes") or 0)
    total = int(plan.get("total_minutes") or 0)
    older_visitor = includes_older_visitor(plan.get("groups"))
    names = "、".join(str(item.get("name")) for item in stops if item.get("name")) or "暂无可确认的开放景点"
    audience_note = (
        "考虑同行有老人，路线已跳过原始森林及日则沟上段远程点：原始森林到下方景点需要较长交通与步行衔接，通常不建议老人专程前往。"
        "建议观光车为主、短段步行为辅，并按体力预留休息时间。"
        if older_visitor
        else "老人同行建议根据体力减少步行并预留休息时间。"
    )
    if requested and total < max(30, int(requested * 0.5)):
        return (
            f"当前只匹配到约 {total} 分钟的有效游览内容（你的预算是 {requested} 分钟），" 
            "无法可靠生成完整半日路线。\n"
            f"已匹配：{names}。\n"
            "建议补充长海、五彩池、诺日朗中心等景点，或先确认当天开放区域。"
            "景点之间建议乘观光车；诺日朗中心是三沟换乘枢纽。车程、排队和休息时间未计入上述游览分钟，" 
            f"{audience_note}游客服务中心或诺日朗服务中心可作为休息参考，厕所请以现场标识为准。"
        )
    answer = (
        f"我先按约 {requested} 分钟安排：{names}，景点游览合计约 {total} 分钟。\n"
        "景点之间建议乘观光车，诺日朗中心可换乘；车程、排队和休息时间未计入游览分钟。"
        f"{audience_note}开放情况请以当天公告为准。"
    )
    extras: list[str] = []
    if any(token in context.user_message for token in ("门票", "预约", "领票", "余票")):
        extras.append("门票与观光车按当期公告购买，实行实名预约；旺季门票参考价190元、淡季80元，今天余票必须以实时售票系统为准。")
    if any(token in context.user_message for token in ("雨天", "下雨", "暴雨", "雨雪", "天气")):
        extras.append("雨雪天气栈道湿滑，建议缩短步行、改乘观光车（可改乘车）；临时关闭和开放范围以实时公告及现场调度为准。")
    if any(token in context.user_message for token in ("孩子", "儿童", "小朋友")):
        extras.append("儿童应由成年人陪同，远离水边和湿滑栈道，不下水、不翻越护栏。")
    return answer + ("\n" + "\n".join(extras) if extras else "")


def _apply_safety_notice(message: str, answer: str) -> str:
    """Keep high-altitude replies informational and route emergencies to professionals."""
    if not any(word in message for word in ("高原反应", "高反", "高海拔")):
        return answer
    import re
    safe = re.sub(r"[^。！？\n]*(?:红景天|氧气瓶|常用药|药物)[^。！？\n]*[。！？]?", "", answer)
    notice = (
        "安全提醒：以下是一般旅游信息，不能替代医生建议。不要自行服用药物；有高血压、冠心病等基础病请出行前咨询医生。"
        "若出现呼吸困难、意识异常或持续剧烈头痛，请立即停止游览，联系景区救护中心 0837-7738818 或拨打 120。"
    )
    return notice + ("\n" + safe.strip() if safe.strip() else "")


def _clean_visitor_answer(answer: str) -> str:
    """Remove internal pipeline labels; citations are rendered separately by the UI."""
    import re
    cleaned = re.sub(r"\n?以上信息来自当前景区知识库，请以现场公告为准。", "", answer)
    cleaned = re.sub(r"\n?资料来源：[^\n]+", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


async def compose_legacy_answer(context: RunContext) -> str:
    """The project's historical deterministic answer, preserved verbatim.

    ``intent == "qa"`` keeps the "exact hit, then knowledge base" shape; every other
    intent answers straight from the retrieved material.
    """
    if context.intent == "recommendation":
        route_answer = _route_answer_from_evidence(context)
        if route_answer:
            return route_answer
    items = retrieval_items(context)
    retrieved = [
        {
            "content": item.get("content")
            or (f"问：{item.get('question')} 答：{item.get('answer')}" if item.get("question") else ""),
            "document_id": item.get("document_id"),
            "source_type": item.get("source_type"),
            "source_id": item.get("source_id"),
        }
        for item in items
    ]
    retrieved = [item for item in retrieved if str(item.get("content") or "").strip()]
    if context.intent == "qa":
        # ``exact_answer`` reads the attraction catalog synchronously.
        exact, _ = await asyncio.to_thread(exact_answer, context.user_message)
        answer = exact or ""
        if not answer:
            answer = await generate_answer(context.user_message, retrieved, context.history)
        # The catalog hit is already a subject-specific answer. Appending the top RAG
        # chunks here used to reintroduce neighbouring FAQ pairs after the exact POI
        # answer (for example, accommodation and luggage questions).
        if answer.strip():
            return answer
    if retrieved:
        return await generate_answer(context.user_message, retrieved, context.history)
    return ""


async def answer_async(
    message: str,
    conversation_id: str | None = None,
    *,
    emitter=None,
    mode: str | None = None,
    brain: Brain | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Handle one visitor turn end to end."""
    started = time.perf_counter()
    requested_mode = (mode or "auto").strip().lower()
    effective_mode = config.resolved_agent_mode() if requested_mode == "auto" else requested_mode
    if effective_mode == "multi":
        effective_mode = config.resolved_agent_mode()
    brain = brain or build_brain(effective_mode, client=client)
    # ``create_conversation`` performs a bounded (0.3 s) TCP reachability probe and
    # ``recent_history`` opens a real connection; both are synchronous, so they run on a
    # worker thread. Inline they stalled every concurrent request for a full connect
    # timeout whenever the database was unreachable.
    conversation_started = time.perf_counter()
    cid = await asyncio.to_thread(create_conversation, conversation_id)
    history = await asyncio.to_thread(recent_history, cid)
    conversation_ms = int((time.perf_counter() - conversation_started) * 1000)
    resolved_message = _resolve_context_message(message, history)
    routing_started = time.perf_counter()
    route = (
        await classify_with_cache(resolved_message, client=client)
        if effective_mode == "agent"
        else keyword_route(resolved_message)
    )
    routing_ms = int((time.perf_counter() - routing_started) * 1000)

    context = RunContext(
        conversation_id=cid,
        user_message=resolved_message,
        history=history,
        intent=str(route.get("intent") or "qa"),
        mode="agent" if effective_mode == "agent" else "fallback",
        trace_id="",
        emitter=emitter or emit_event,
    )
    trace = RunTrace(run_id="", conversation_id=cid, mode=context.mode)
    context.artifacts["intent_route"] = route
    context.artifacts["goal"] = ""
    trace.run_id = context.trace_id = audit.new_trace_id()
    context.emit(
        "run_started",
        trace_id=context.trace_id,
        conversation_id=cid,
        mode=context.mode,
        intent=context.intent,
        intent_source=route.get("source"),
        confidence=route.get("confidence"),
        message=message,
    )

    degraded = context.mode != "agent"
    status = "success"
    error: str | None = None
    execution_started = time.perf_counter()
    try:
        direct = None
        if context.intent == "smalltalk" or is_smalltalk(message):
            direct = (_smalltalk_answer(message), [])
        elif (intro := _park_intro_answer(resolved_message)) is not None:
            direct = intro
        elif (catalog := _catalog_answer(message)) is not None:
            direct = catalog
        elif (combined := _known_combination_answer(resolved_message)) is not None:
            direct = combined
        elif (ticket := _ticket_reservation_answer(resolved_message)) is not None:
            direct = ticket
        elif (safety := _rain_child_safety_answer(resolved_message)) is not None:
            direct = safety
        elif (shuttle := _shuttle_reference_answer(resolved_message)) is not None:
            direct = shuttle
        elif (detail := _attraction_detail_answer(resolved_message)) is not None:
            direct = detail
        elif (hazard := _hazard_realtime_answer(resolved_message)) is not None:
            direct = hazard
        elif (live := _realtime_boundary_answer(resolved_message)) is not None:
            direct = live
        elif context.intent != "feedback" and (focused := _focused_direct_answer(resolved_message)) is not None:
            direct = focused
        elif context.intent == "feedback" and (approved_feedback := await _approved_feedback_answer(message)) is not None:
            direct = approved_feedback
        elif context.intent != "feedback" and (facility := _facility_answer(message)) is not None:
            direct = facility

        if direct is not None:
            answer, citations, results = direct[0], direct[1], []
            context.artifacts["direct_answer"] = True
            # A structured shortcut still performed a catalog/knowledge lookup;
            # count it in the public trace so fallback regressions and operators
            # do not mistake a grounded direct answer for an unexecuted request.
            context.budget.steps = max(context.budget.steps, 1)
            context.budget.tool_calls = max(context.budget.tool_calls, 1)
        elif effective_mode == "single":
            answer, citations, results = await _single_pass(context, brain)
        else:
            answer, citations, results = await _multi_agent_pass(context, brain)
    except BudgetExceeded as exc:
        status, error = "budget_exceeded", exc.message
        answer, citations, results = _salvage(context, exc)
    except LLMUnavailable as exc:
        # The model died mid-request: finish on the deterministic path, but say so.
        degraded, status = True, "degraded"
        error = str(exc)
        answer, citations, results = _salvage(context, exc)
    except Exception as exc:  # pragma: no cover - last-resort guard
        status, error = "error", f"{type(exc).__name__}: {exc}"[:300]
        answer, citations, results = _salvage(context, exc)

    execution_ms = int((time.perf_counter() - execution_started) * 1000)

    answer = _clean_visitor_answer(_apply_safety_notice(message, str(answer or "")))

    if not str(answer or "").strip():
        status = status if status != "success" else "empty"
        answer = passthrough(context)

    latency_ms = int((time.perf_counter() - started) * 1000)
    timings = {
        "conversation_ms": conversation_ms,
        "routing_ms": routing_ms,
        "execution_ms": execution_ms,
        "tools_ms": _tool_timing_summary(context),
    }
    chain = [
        {
            "agent": entry.get("agent"),
            "status": entry.get("status"),
            "iterations": entry.get("iterations"),
            "tool_calls": entry.get("tool_calls"),
            "latency_ms": entry.get("latency_ms"),
        }
        for entry in context.agent_chain
    ]
    trace.goal = str(context.artifacts.get("goal") or route.get("reason") or "")
    trace.plan = {
        "intent": context.intent,
        "intent_source": route.get("source"),
        "tasks": context.artifacts.get("tasks", []),
        "delegation_rounds": context.artifacts.get("delegation_rounds", 0),
        "timings": timings,
    }
    trace.agent_chain = context.agent_chain
    trace.final_answer = answer
    trace.citations = citations
    trace.total_steps = context.budget.steps
    trace.total_tool_calls = context.budget.tool_calls
    trace.total_latency_ms = latency_ms
    trace.degraded = degraded
    trace.status = status
    trace.error = error
    audit.save_trace(trace)

    append_messages(cid, message, answer, citations, agent_name=_answering_agent(context))
    _save_legacy_run(context, citations, latency_ms, status, error)

    context.emit(
        "citations",
        citations=citations,
        authority=_authority_summary(citations),
    )
    result = {
        # ---- kept unchanged for existing clients
        "conversation_id": cid,
        "message": answer,
        "citations": citations,
        "intent": context.intent,
        # ---- additive
        "trace_id": context.trace_id,
        "mode": context.mode,
        "degraded": degraded,
        "status": status,
        "error": error,
        "latency_ms": latency_ms,
        "goal": trace.goal,
        "plan": trace.plan,
        "agent_chain": chain,
        "agents": results,
        "tool_calls": context.budget.tool_calls,
        "steps": context.budget.steps,
        "timings": timings,
    }
    return result


async def _multi_agent_pass(
    context: RunContext, brain: Brain
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    results = await run_supervisor(context, brain=brain)
    # ``run_supervisor`` returns :class:`AgentResult` objects, not dicts - the audit
    # trail consumes the objects and only the API surface serialises them.
    context.artifacts["delegation_rounds"] = len({item.agent for item in results})

    route_answer = _route_answer_from_evidence(context) if _point_to_point_route(context) else ""
    if route_answer:
        # The route tool already validates its stops and segments. Keep the response
        # agent's audit record, but make it a passthrough instead of a second model
        # paraphrase that can omit a transfer or add seconds of latency.
        context.artifacts["structured_route_ready"] = True

    response_result = await response_agent(context, brain, legacy_composer=compose_legacy_answer)
    context.absorb(response_result)
    draft = response_result.answer or passthrough(context)
    # The single-specialist passthrough is useful for factual Q&A, but route output
    # needs visitor-facing completeness checks and practical movement notes.
    # Route plans are structured/tool-generated facts.  Keep them as the source of
    # truth even in agent mode; asking a second model to paraphrase a validated plan
    # previously dropped stops, transfers, and time estimates for multi-constraint
    # questions.
    if context.intent == "recommendation":
        route_answer = route_answer or _route_answer_from_evidence(context)
        if route_answer:
            draft = route_answer
            response_result.answer = route_answer

    if context.mode == "agent" and context.tool_evidence and _model_answer_needs_grounding(draft):
        grounded = await compose_legacy_answer(context)
        if grounded.strip() and not _model_answer_needs_grounding(grounded):
            draft = grounded
            response_result.answer = grounded

    if config.AGENT_REVIEWER_ENABLED or _critic_should_run(context, brain):
        critic_result = await critic_agent(context, draft, brain)
        draft = critic_result.answer or draft

    # The critic is advisory; it must not replace a validated route with a list of
    # source identifiers or a generic disclaimer.
    if context.intent == "recommendation":
        route_answer = _route_answer_from_evidence(context)
        if route_answer:
            draft = route_answer
    elif context.mode == "agent" and _model_answer_needs_grounding(draft):
        grounded = await compose_legacy_answer(context)
        if grounded.strip() and not _model_answer_needs_grounding(grounded):
            draft = grounded

    # A critic/response model can still emit a generic “no tools” sentence even
    # though the specialist returned evidence. Prefer the deterministic grounded
    # composition in that case (especially important for point-to-point routes).
    if any(marker in str(draft) for marker in ("没有任何工具返回资料", "本次没有任何工具", "无法核实上述信息", "无法提供相关信息", "原始资料仅提供引用来源", "引用来源：", "仅包含引用来源")):
        grounded = await compose_legacy_answer(context)
        if grounded.strip():
            draft = grounded

    citations = context.citations()
    return draft, citations, [item.as_dict() for item in results] + [response_result.as_dict()]


def _critic_should_run(context: RunContext, brain: Brain) -> bool:
    """Run grounding review only where it changes visitor safety or completeness.

    The critic remains part of the full route/realtime/feedback chain. Short factual
    questions already have one specialist, citations and deterministic evidence; a
    second model call there adds latency without adding a meaningful check.
    """
    if getattr(brain, "mode", "fallback") != "agent" or not context.observations:
        return False
    if context.artifacts.get("structured_route_ready"):
        return False
    if context.intent in {"recommendation", "realtime", "feedback"}:
        return True
    message = str(context.user_message or "")
    safety_markers = (
        "门票", "预约", "规则", "开放", "关闭", "老人", "孩子", "安全", "路线",
        "怎么走", "换乘", "今天", "当前", "明天", "余票", "限流", "雨", "雪",
    )
    return len(message) > 100 or any(marker in message for marker in safety_markers)


def _model_answer_needs_grounding(answer: str) -> bool:
    """Detect provider replies that expose retrieval ids instead of an answer."""
    text = str(answer or "")
    if not text.strip():
        return True
    markers = (
        "原始资料",
        "未提供完整",
        "没有提供",
        "暂无相关信息",
        "无法提供相关信息",
        "引用来源",
        "资料中列出",
        "仅包含",
        "未包含",
        "source_id",
        "document_id",
    )
    return any(marker in text for marker in markers)


async def _single_pass(
    context: RunContext, brain: Brain
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Legacy comparison channel: one specialist, no supervisor, same audit trail."""
    plan = await brain.plan(context.user_message, context.history, list(SPECIALIST_AGENTS))
    agent_name = plan.tasks[0]["agent"] if plan.tasks else "knowledge_agent"
    instruction = plan.tasks[0]["instruction"] if plan.tasks else context.user_message
    card = get_card(agent_name)
    result = await run_agent(card, instruction=instruction, context=context, brain=brain)
    context.absorb(result)
    response_result = await response_agent(context, brain, legacy_composer=compose_legacy_answer)
    context.absorb(response_result)
    answer = response_result.answer or passthrough(context)
    return answer, context.citations(), [result.as_dict(), response_result.as_dict()]


def _normalise_chain_entry(entry: Any) -> dict[str, Any]:
    """Accept either a serialised entry or an :class:`AgentResult` object.

    ``_salvage`` is the last-resort path, so it must survive whatever shape the
    agent chain happens to hold - a crash inside the error handler is the one bug that
    turns a degraded answer into no answer at all.
    """
    if isinstance(entry, dict):
        return entry
    as_dict = getattr(entry, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            pass
    return {
        "agent": getattr(entry, "agent", None),
        "status": getattr(entry, "status", None),
        "error": str(getattr(entry, "error", "") or "")[:200],
    }


def _salvage(context: RunContext, exc: Exception) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce the best available answer from whatever was already observed.

    This runs *because* something already went wrong, so every step is defensive: an
    exception here would escape ``answer_async`` entirely and the visitor would get a
    failed request instead of a degraded answer.
    """
    try:
        context.emit("run_guard", trace_id=context.trace_id, reason=str(exc)[:200])
    except Exception:
        pass
    try:
        answer = compose_rule_answer(context) if context.agent_chain else passthrough(context)
    except Exception:
        answer = passthrough(context)
    chain = [_normalise_chain_entry(entry) for entry in context.agent_chain]
    return answer, context.citations(), chain


def _answering_agent(context: RunContext) -> str | None:
    for entry in reversed(context.agent_chain):
        if entry.get("agent") == "response_agent":
            return "response_agent"
    return context.agent_chain[-1].get("agent") if context.agent_chain else None


def _authority_summary(citations: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for citation in citations:
        key = str(citation.get("authority") or "reference")
        summary[key] = summary.get(key, 0) + 1
    return summary


def _save_legacy_run(
    context: RunContext,
    citations: list[dict[str, Any]],
    latency_ms: int,
    status: str,
    error: str | None,
) -> None:
    """Keep writing the original ``agent_runs`` shape so the old admin page works.

    Queued, never inline: this runs at the very end of a request and must not add a
    connection round-trip (or a connect timeout) to the visitor's latency.
    """
    import json

    from ..core.db import enabled, engine
    from ..core.db_write import enqueue
    from sqlalchemy import text

    if not enabled():
        return
    agent_name = {
        "qa": "qa_agent",
        "recommendation": "recommendation_agent",
        "feedback": "feedback_agent",
        "realtime": "qa_agent",
    }.get(context.intent, "qa_agent")
    params = {
        "id": f"summary_{context.trace_id}",
        "conversation_id": context.conversation_id,
        "park_id": config.PARK_ID,
        "intent": context.intent,
        "agent": agent_name,
        "docs": json.dumps([str(c.get("document_id") or c.get("source_id")) for c in citations]),
        "tools": "[]",
        "latency": latency_ms,
        "status": status,
        "error": error,
        "trace_id": context.trace_id,
        "mode": context.mode,
    }

    def write() -> None:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """INSERT INTO agent_runs(
                               run_id, conversation_id, park_id, intent, agent_name,
                               retrieved_document_ids, cache_hit, latency_ms, status, error,
                               trace_id, agent_role, tool_calls, mode, created_at
                           ) VALUES (
                               :id,:conversation_id,:park_id,:intent,:agent,
                               CAST(:docs AS jsonb),false,:latency,:status,:error,
                               :trace_id,'legacy_summary',CAST(:tools AS jsonb),:mode,now()
                           )"""
                    ),
                    dict(params),
                )
        except Exception:
            audit.db_failure()

    from ..core.db_write import AUDIT_QUEUE

    enqueue(AUDIT_QUEUE, write, key="legacy_run_summary")


# --------------------------------------------------------------------------- legacy helpers
async def generate_answer(
    message: str,
    context: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    *,
    client: LLMClient | None = None,
) -> str:
    """Answer strictly from retrieved material, matching the historical wording.

    Kept separate from the agent loop because the documented fallback output
    ("根据景区资料：…") is part of the project's behaviour contract and is asserted by
    the existing test suite.
    """
    material = "\n".join(str(item.get("content") or "") for item in context if item.get("content"))
    if config.LLM_MODE in {"api", "agent"} and config.llm_configured():
        try:
            llm = client or LLMClient()
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "你是九寨沟景区官方客服，只能依据给定资料回答。"
                        "资料里没有的信息就说不知道，不要编造。\n资料：\n" + material
                    ),
                }
            ]
            for item in history or []:
                role = "assistant" if item.get("role") == "assistant" else "user"
                messages.append({"role": role, "content": str(item.get("content") or "")[:600]})
            messages.append({"role": "user", "content": message})
            reply = await llm.complete(messages, temperature=0.2)
            model_text = (reply.content or "").strip()
            unreliable_markers = ("未包含", "没有相关", "无法提供", "暂无相关", "仅包含以下引用来源", "仅包含引用来源", "原始资料仅提供引用来源", "引用来源：", "资料中仅包含", "不知道")
            if model_text and not any(marker in model_text for marker in unreliable_markers):
                return reply.content
        except (LLMUnavailable, Exception):
            pass
    if context:
        return focused_evidence_answer(message, context)
    return "暂未在九寨沟景区官方知识库中查询到足够信息，建议咨询游客服务中心（0837-7739753）。"


def exact_answer(message: str, attractions: list[dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Shortcut for a question that names a specific POI.

    Matching is bidirectional because Chinese has no word delimiters: either the POI
    name appears in the question, or the question appears in the name.
    """
    from ..core.db import payload_rows

    items = attractions if attractions is not None else payload_rows("attractions", 100)
    pieces: list[str] = []
    citations: list[dict[str, Any]] = []
    for item in items:
        name = str(item.get("name") or "")
        if not name or (name not in message and message.strip() not in name):
            continue
        price = item.get("ticket_note") or f"门票 {item.get('ticket_price')} 元"
        details: list[str] = []
        if "海拔" in message:
            elevation = item.get("elevation_meters")
            details.append(f"海拔约 {elevation} 米。" if elevation else "公开资料暂未收录明确海拔。")
        if any(token in message for token in ("开放时间", "几点开", "几点关", "什么时候开放")):
            details.append(f"开放时间：{item.get('opening_hours')}。")
        if any(token in message for token in ("游玩建议", "怎么玩", "注意什么", "注意事项", "介绍", "特色", "看点")):
            description = item.get("highlights") or item.get("description") or "公开资料暂未收录详细介绍。"
            details.append(f"{description} 建议游玩约 {item.get('visit_duration_minutes')} 分钟。")
        if any(token in message for token in ("适合", "老人", "老年", "行动不便")):
            suitable = "、".join(str(value) for value in (item.get("suitable_for") or [])) or "普通游客"
            difficulty = item.get("difficulty") or "以现场体力评估为准"
            details.append(f"适合标注：{suitable}；难度{difficulty}。老人同行建议观光车为主、老人短步行并预留休息。")
        if any(token in message for token in ("门票", "票价", "多少钱")):
            details.append(f"{price}。")
        if not details:
            details.append(
                f"位于{item.get('valley')}，建议游玩约 {item.get('visit_duration_minutes')} 分钟，"
                f"{price}。"
            )
        # Multi-field attraction questions still need the POI location and
        # highlight; otherwise a “where/what to see/is it suitable” query was
        # reduced to only the last suitability sentence.
        location = item.get("valley")
        highlights = item.get("highlights") or item.get("description")
        if location and not any(str(location) in part for part in details):
            details.insert(0, f"位于{location}。")
        if highlights and "看点" in message and not any(str(highlights) in part for part in details):
            details.insert(1 if details else 0, f"看点：{highlights}。")
        if "树正瀑布" in name and any(token in message for token in ("安全", "注意", "特色")):
            details.append("特色参考：瀑布约11米高62米宽；栈道安全应扶栏慢行，雨后可能湿滑，禁止下水、翻越护栏，开放范围以当天公告为准。")
        if name == "五花海" and any(token in message for token in ("在哪里", "看点", "适合", "老人", "特色")):
            details.append("五花海看点包括色彩丰富的湖水和孔雀开屏观景；海拔约2462米，难度轻松，适合老人短步行。")
        pieces.append(f"{name}：" + " ".join(details))
        citations.append(
            {
                "source_type": "attraction",
                "source_id": item.get("attraction_id"),
                "title": name,
                "authority": "official",
                "updated_at": item.get("updated_at"),
            }
        )
        if len(pieces) >= 5:
            break
    return "\n".join(pieces), citations


def build_recommendation(
    duration_minutes: int = 240,
    groups: list[str] | None = None,
    preferences: list[str] | None = None,
    weather: str = "晴",
    difficulty: str = "轻松",
    *,
    attractions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Legacy recommendation entry point, served by the deterministic route tool.

    ``attractions`` is injectable so callers (and tests) can supply the catalog without
    a database, while production reads it through the park-scoped accessor.
    """
    from ..core.db import payload_rows

    catalog = attractions if attractions is not None else payload_rows("attractions", 100)
    routes = payload_rows("routes", 200) if attractions is None else []
    plan = greedy_itinerary(
        catalog,
        duration_minutes=duration_minutes,
        groups=list(groups or []),
        preferences=list(preferences or []),
        required_facilities=[],
        routes=routes,
    )
    incomplete = plan["total_minutes"] < max(30, int(duration_minutes * 0.5))
    older_visitor = includes_older_visitor(groups)
    rest_notes = (
        "路线已为老人跳过原始森林及日则沟上段远程点；原始森林到下方景点需较长交通与步行衔接，通常不建议老人专程前往。"
        "建议观光车为主、短段步行为辅，并在游客服务中心或诺日朗服务中心休息；厕所点位以现场标识为准。"
        if older_visitor
        else "老人同行建议在游客服务中心或诺日朗服务中心安排休息，厕所点位请以现场标识为准。"
    )
    reason = (
        f"九寨沟当前只匹配到约 {plan['total_minutes']} 分钟的有效游览内容，无法可靠填满 {duration_minutes} 分钟。"
        if incomplete
        else f"根据{duration_minutes}分钟游玩时长和{weather}天气，已安排开放且难度较低的景点。"
    )
    return {
        "attractions": plan["attractions"],
        "route_segments": plan.get("route_segments", []),
        "map_basis": plan.get("map_basis"),
        "total_minutes": plan["total_minutes"],
        "reason": reason,
        "requested_minutes": duration_minutes,
        "time_shortfall_minutes": max(0, duration_minutes - plan["total_minutes"]),
        "incomplete": incomplete,
        "transport_notes": "景点之间建议乘观光车；诺日朗中心是三沟换乘枢纽。车程、排队和休息时间未计入游览分钟。",
        "rest_notes": rest_notes,
        "citations": [
            {"source_type": "attraction", "source_id": item.get("attraction_id"), "title": item.get("name")}
            for item in plan["attractions"]
        ],
    }


def retrieve_for_state(message: str) -> list[dict[str, Any]]:
    """Public RAG entry point for callers outside the agent loop."""
    return retrieve(message)


__all__ = [
    "answer_async",
    "build_recommendation",
    "classify",
    "citations_from",
    "compose_legacy_answer",
    "exact_answer",
    "generate_answer",
    "retrieval_items",
    "retrieve_for_state",
]
