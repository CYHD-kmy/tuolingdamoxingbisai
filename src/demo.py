"""
演示模式 — 用样本数据构造完整的 PipelineState，跳过网络和 LLM 调用。

用于:
  - 开发调试 (无需真实数据和 API Key)
  - 看板效果展示
  - CI / 集成测试

使用方式:
    python -m src.main --demo
"""

from __future__ import annotations

from datetime import datetime

from .graph.state import PipelineState
from .screening.scorer import FactorScore
from .agents.base import AnalystReport
from .agents.models import (
    DebateResult, DebateRound, ResearchVerdict,
    PositionLimit, FinalDecision, PortfolioResult,
)

# ── 样本股票池 ──────────────────────────────────

_SAMPLE_STOCKS = [
    ("600519", "贵州茅台", 1680.50, 2.35, 1.42, 3.2, 4.8e9, 28.5, 2.1e12),
    ("000858", "五粮液",  152.30,  1.88, 1.18, 2.5, 2.1e9, 22.3, 5.9e11),
    ("300750", "宁德时代", 205.60, 3.12, 1.95, 4.8, 5.2e9, 35.6, 9.0e11),
    ("002594", "比亚迪",   268.00, 4.55, 2.30, 6.1, 7.8e9, 42.0, 7.8e11),
    ("601318", "中国平安",  48.20,  0.65, 0.88, 1.2, 3.5e9, 10.5, 8.8e11),
    ("000333", "美的集团",  62.80,  1.20, 1.05, 2.0, 2.8e9, 15.2, 4.5e11),
]


def generate_demo_state() -> PipelineState:
    """生成包含完整演示数据的 PipelineState"""

    state = PipelineState(total_capital=500_000.0)
    today = datetime.now()

    # ── 阶段 1: 候选池 (多因子打分) ──────────────
    state.candidates = _make_candidates()
    state.daily_data = _make_daily_data()
    state.fund_flows = _make_fund_flows()

    # ── 阶段 2: 深度分析 ─────────────────────────
    state.analyst_reports = _make_analyst_reports()
    state.debates = _make_debates()
    state.verdicts = _make_verdicts()

    # ── 阶段 3: 风控 ─────────────────────────────
    state.position_limits = _make_position_limits()

    # ── 阶段 4: 组合构建 ─────────────────────────
    state.final_result = _make_portfolio_result()

    # ── 元信息 ───────────────────────────────────
    state.date = today.strftime("%Y-%m-%d")
    state.stage = "done"
    state.elapsed = {
        "screening": 0.8,
        "analysis": 12.3,
        "risk": 0.4,
        "portfolio": 2.1,
    }

    return state


# ── 候选池 ──────────────────────────────────────

def _make_candidates() -> list[FactorScore]:
    """生成 6 只样本股票的多因子得分"""
    return [
        FactorScore(code="600519", name="贵州茅台", composite=88.5, scores={
            "trend": 90, "momentum": 72, "volume_price": 85,
            "capital_flow": 92, "northbound": 88, "sentiment": 80,
            "quality": 85, "risk": 78, "liquidity": 95, "shareholder_conc": 75,
        }),
        FactorScore(code="000858", name="五粮液", composite=82.3, scores={
            "trend": 85, "momentum": 68, "volume_price": 78,
            "capital_flow": 88, "northbound": 82, "sentiment": 75,
            "quality": 82, "risk": 80, "liquidity": 90, "shareholder_conc": 70,
        }),
        FactorScore(code="300750", name="宁德时代", composite=79.8, scores={
            "trend": 78, "momentum": 82, "volume_price": 88,
            "capital_flow": 85, "northbound": 90, "sentiment": 72,
            "quality": 72, "risk": 60, "liquidity": 88, "shareholder_conc": 65,
        }),
        FactorScore(code="002594", name="比亚迪", composite=76.2, scores={
            "trend": 82, "momentum": 90, "volume_price": 80,
            "capital_flow": 78, "northbound": 75, "sentiment": 75,
            "quality": 68, "risk": 55, "liquidity": 82, "shareholder_conc": 60,
        }),
        FactorScore(code="601318", name="中国平安", composite=71.5, scores={
            "trend": 60, "momentum": 55, "volume_price": 62,
            "capital_flow": 75, "northbound": 70, "sentiment": 68,
            "quality": 80, "risk": 85, "liquidity": 92, "shareholder_conc": 72,
        }),
        FactorScore(code="000333", name="美的集团", composite=68.4, scores={
            "trend": 75, "momentum": 65, "volume_price": 70,
            "capital_flow": 65, "northbound": 80, "sentiment": 62,
            "quality": 76, "risk": 82, "liquidity": 75, "shareholder_conc": 78,
        }),
    ]


# ── 日线数据 (简化版: 最近 5 日) ────────────────

def _make_daily_data() -> dict[str, list]:
    """为每只样本股生成 20 天 OHLCV + 指标的模拟数据"""
    from .data.fetchers.akshare_fetcher import StockDaily

    data: dict[str, list] = {}
    for code, name, price, pct, _vr, turnover, amount, _pe, _mv in _SAMPLE_STOCKS:
        days = []
        base = price * 0.88  # 20天前大约 -12%
        for i in range(20):
            d = StockDaily(
                date=f"2026-05-{i+1:02d}",
                open=round(base, 2),
                high=round(base * 1.02, 2),
                low=round(base * 0.98, 2),
                close=round(base * 1.01, 2),
                volume=2e7,
                amount=base * 2e7 * 0.7,
                pct_chg=round(pct * (0.5 + i / 20), 2),
                turnover=round(turnover, 2),
                ma5=round(base * 0.99, 2),
                ma10=round(base * 0.97, 2),
                ma20=round(base * 0.95, 2),
                macd_dif=round(0.5 + i * 0.1, 3),
                macd_dea=round(0.3 + i * 0.08, 3),
                macd_bar=round(0.2 + i * 0.02, 3),
                rsi_6=round(55 + i * 0.5, 2),
                rsi_14=round(52 + i * 0.3, 2),
            )
            base *= 1.006
            days.append(d)
        data[code] = days
    return data


# ── 资金流向 ────────────────────────────────────

def _make_fund_flows() -> dict[str, list[FundFlow]]:
    """为每只样本股生成 5 天资金流向数据"""
    from .data.fetchers.akshare_fetcher import FundFlow

    flows: dict[str, list] = {}
    flow_configs = {
        "600519": (+800, +1200, +450, +2100, +950),
        "000858": (+500, +650, -200, +880, +720),
        "300750": (+1500, -300, +2200, +800, +1800),
        "002594": (+2000, +3500, +1200, +2800, +4100),
        "601318": (-500, +300, +150, +600, +200),
        "000333": (+150, +400, -100, +250, +350),
    }
    for code, name, *_ in _SAMPLE_STOCKS:
        cfg = flow_configs.get(code, (+100, +100, +100, +100, +100))
        code_flows = []
        for i, main_net in enumerate(cfg):
            code_flows.append(FundFlow(
                date=f"2026-05-{21+i:02d}",
                main_net_inflow=main_net,
                super_large_net=round(main_net * 0.5, 1),
                large_net=round(main_net * 0.3, 1),
                medium_net=round(main_net * 0.1, 1),
                small_net=round(main_net * 0.1, 1),
                main_pct=round(abs(main_net) / 50000 * 100, 1),
            ))
        flows[code] = code_flows
    return flows


# ── 分析师报告 ──────────────────────────────────

def _make_analyst_reports() -> dict[str, list[AnalystReport]]:
    """为每只股票生成四维分析师报告"""
    reports_map: dict[str, list[AnalystReport]] = {}

    templates = {
        "600519": {
            "technical": ("bullish", 0.82, "均线多头排列，MACD金叉后持续上行，RSI处于强势区间未超买。"
                         "近5日量价配合良好，放量突破前高阻力位，短期上行趋势明确。"),
            "fundamentals": ("bullish", 0.78, "PE 28.5倍处于近3年低位，ROE维持30%+，Q1营收同比+15%。"
                             "品牌护城河深厚，直销占比提升带动毛利率改善，估值有修复空间。"),
            "fund_flow": ("bullish", 0.85, "近5日主力资金持续净流入累计超过5.5亿，超大单占比显著提升。"
                          "北向资金连续3日增持，融资余额温和上升，资金面偏多。"),
            "news": ("bullish", 0.72, "飞天茅台批价企稳回升，i茅台平台GMV增长超预期。"
                               "机构研报普遍看好，市场情绪偏积极。近期无重大利空。"),
        },
        "000858": {
            "technical": ("bullish", 0.75, "股价站上所有均线，成交量温和放大。MACD柱状线转正，"
                         "KDJ金叉向上。短期压力位158元附近，若能放量突破则打开上行空间。"),
            "fundamentals": ("bullish", 0.80, "PE 22.3倍，低于白酒板块均值。Q1净利润同比+18%，"
                             "产品结构升级顺利，系列酒增速超预期。分红率3.2%具有防御价值。"),
            "fund_flow": ("bullish", 0.78, "近5日主力净流入累计3.5亿，大单持续买入。"
                          "融资买入额上升，市场关注度提升。但北向有小幅流出需关注。"),
            "news": ("neutral", 0.55, "白酒板块整体回暖，但消费税改革传闻有不确定性。"
                               "渠道库存处于健康水平，中秋国庆备货预期积极。"),
        },
        "300750": {
            "technical": ("bullish", 0.80, "股价突破60日均线压制，MACD零轴上方金叉。"
                         "成交量较前一周放大60%，量价共振。近期有望挑战220元前高。"),
            "fundamentals": ("bullish", 0.70, "PE 35.6倍略高但增长可期。全球份额持续提升，"
                             "神行电池量产进度超预期。储能业务增速80%+，第二增长曲线确立。"),
            "fund_flow": ("bullish", 0.82, "近5日主力净流入超6亿，超大单买入积极。"
                          "北向资金本周净买入3.2亿，机构配置需求旺盛。"),
            "news": ("bullish", 0.75, "锂电池行业排产超预期，上游碳酸锂价格企稳。"
                               "公司与多家车企签订长期供货协议，市场信心增强。"),
        },
        "002594": {
            "technical": ("bullish", 0.88, "日线走出V型反转，连续5日阳线收盘。MACD红柱持续放大，"
                         "RSI强势但未超买。量价齐升，突破250元关键压力位。短期动能强劲。"),
            "fundamentals": ("bullish", 0.75, "PE 42倍偏高，但Q1新能源车销量同比+60%，市占率超35%。"
                             "海外出口翻倍增长，高端品牌仰望交付超预期。规模效应持续释放。"),
            "fund_flow": ("bullish", 0.90, "近5日主力净流入超14亿，为全市场最高。"
                          "游资和机构共同买入，龙虎榜显示多家知名营业部大额买入。"),
            "news": ("bullish", 0.85, "新能源汽车政策持续加码，以旧换新补贴落地。"
                               "比亚迪海鸥出口欧洲获认证通过，海外市场加速扩张。"),
        },
        "601318": {
            "technical": ("neutral", 0.55, "股价横盘震荡，均线粘合方向不明。MACD零轴附近徘徊，"
                         "成交量萎缩。需等待方向选择，短期缺乏明确交易信号。"),
            "fundamentals": ("bullish", 0.72, "PE 10.5倍处于历史低位，内含价值增长稳健。"
                             "NBV增速转正，代理人改革初见成效。分红率5.2%提供安全边际。"),
            "fund_flow": ("neutral", 0.58, "主力资金小幅净流入但规模有限。北向资金有进有出，"
                          "整体资金面中性。需要催化剂打破僵局。"),
            "news": ("neutral", 0.50, "保险板块整体走势平淡。利率下行对利差有负面影响，"
                               "但权益市场回暖利好投资收益。短期缺乏明确催化剂。"),
        },
        "000333": {
            "technical": ("bullish", 0.68, "股价沿20日均线稳步上行，均线呈现多头排列初期。"
                         "MACD即将金叉，若能放量则确认上行趋势。短期目标65元。"),
            "fundamentals": ("bullish", 0.76, "PE 15.2倍合理偏低，Q1营收同比+12%超预期。"
                             "海外业务占比提升至40%+，OBM自主品牌增长强劲。智能家居生态完善。"),
            "fund_flow": ("neutral", 0.55, "主力资金小幅流入约1000万，方向不明。"
                          "近期以散户资金为主，机构持仓相对稳定。等待业绩催化。"),
            "news": ("neutral", 0.52, "家电以旧换新政策利好但效果待观察。"
                               "海外关税不确定性形成压制。公司回购计划提供底部支撑。"),
        },
    }

    for code, name, *_ in _SAMPLE_STOCKS:
        tmpl = templates.get(code, {})
        reports = []
        for atype in ["technical", "fundamentals", "fund_flow", "news"]:
            t = tmpl.get(atype, ("neutral", 0.50, f"{name} {atype} 分析暂无明确信号。"))
            reports.append(AnalystReport(
                analyst_type=atype,
                code=code, name=name,
                signal=t[0], confidence=t[1], reasoning=t[2],
                key_points=[t[2][:60]],
                risks=["市场整体下行风险", "行业政策不确定性"],
            ))
        reports_map[code] = reports
    return reports_map


# ── 辩论 ────────────────────────────────────────

def _make_debates() -> dict[str, DebateResult]:
    """生成辩论记录"""
    debates: dict[str, DebateResult] = {}

    debate_data = {
        "600519": [
            ("茅台品牌壁垒深厚，直销占比提升持续改善毛利率，当前估值处于历史低位，"
             "北向资金持续流入，技术形态多头排列，建议积极买入。",
             "白酒行业整体增速放缓，年轻人饮酒习惯变化构成长期隐忧。"
             "当前价位距离历史高点仍有距离，且消费税改革可能带来政策风险。",
             "短期消费复苏趋势明确，中秋国庆备货季即将启动，批价企稳回升。"
             "即使行业增速放缓，茅台作为龙头将挤压中小酒企份额实现超越行业增长。",
             "综合来看多头逻辑占优，但需关注消费税政策和批价波动。"
             "建议在1650-1720区间分批建仓，设5%止损线。"),
        ],
        "300750": [
            ("全球动力电池龙头地位稳固，储能第二曲线增速80%+，神行电池量产带来新增量。"
             "技术面突破60日线压制，量价配合良好，是典型的右侧买点。",
             "锂电行业产能过剩隐忧未消，碳酸锂价格仍有下行压力。"
             "海外政策风险增加，欧美对中国电池产业链设限可能影响出口。",
             "产能过剩主要集中在低端产品，宁德在高端电池市场议价能力强。"
             "海外建厂策略有效规避关税壁垒，长期成长逻辑不变。",
             "多头占优但需控制仓位。行业竞争加剧是中期风险，建议以20%仓位介入。"),
        ],
        "002594": [
            ("比亚迪新能源车销量和市占率持续超预期，海外出口翻倍增长。"
             "技术面V型反转量价齐升，主力资金大规模流入，是当前最强标的。",
             "股价短期涨幅过大，追高风险显著。PE 42倍不便宜，且新能源车行业竞争白热化，"
             "特斯拉降价可能引发价格战，影响利润率。",
             "比亚迪已建立起全产业链成本优势，价格战正是其扩大份额的时机。"
             "高端品牌仰望提升整体毛利率，智能化布局加速。短期强势趋势明确。",
             "短线趋势强劲但估值偏高，建议轻仓参与不宜重仓。回调至240元以下可加仓。"),
        ],
    }

    for code, name, *_ in _SAMPLE_STOCKS:
        rounds = debate_data.get(code)
        if rounds is None:
            debates[code] = DebateResult(code=code, name=name, total_rounds=0)
            continue
        debate_rounds = []
        for i, (bull, bear, rebuttal, summary) in enumerate(rounds, 1):
            debate_rounds.append(DebateRound(
                round_num=i,
                bull_argument=bull,
                bear_argument=bear,
                bull_rebuttal=rebuttal if i >= 2 else "",
                bear_summary=summary,
            ))
        debates[code] = DebateResult(
            code=code, name=name,
            rounds=debate_rounds,
            total_rounds=len(debate_rounds),
        )
    return debates


# ── 研究主管研判 ────────────────────────────────

def _make_verdicts() -> dict[str, ResearchVerdict]:
    """生成研究主管研判结论"""
    return {
        "600519": ResearchVerdict(
            code="600519", name="贵州茅台", direction="buy",
            confidence=0.80, target_price=1780.00, risk_level="low",
            core_reasoning="四维分析一致看多。技术面多头排列，基本面估值低位且Q1超预期，"
                           "资金面北向持续流入，情绪面批价企稳。风险收益比优异。",
            key_risks=["消费税改革不确定性", "白酒行业长期增速放缓"],
        ),
        "000858": ResearchVerdict(
            code="000858", name="五粮液", direction="buy",
            confidence=0.72, target_price=165.00, risk_level="low",
            core_reasoning="估值低于板块均值，Q1业绩超预期。技术面站上所有均线，"
                           "资金面主力持续流入。作为白酒老二具备跟涨茅台的弹性。",
            key_risks=["行业竞争加剧", "消费税政策风险"],
        ),
        "300750": ResearchVerdict(
            code="300750", name="宁德时代", direction="buy",
            confidence=0.75, target_price=225.00, risk_level="medium",
            core_reasoning="技术面和资金面信号强劲，储能第二曲线确立。但估值偏高，"
                           "海外政策风险需关注。建议控制仓位参与。",
            key_risks=["海外贸易壁垒", "行业产能过剩", "碳酸锂价格波动"],
        ),
        "002594": ResearchVerdict(
            code="002594", name="比亚迪", direction="buy",
            confidence=0.78, target_price=290.00, risk_level="medium",
            core_reasoning="四维信号极强，主力资金流入全市场第一。短期动能充足，"
                           "但估值偏高且涨幅已大，追高风险需警惕。建议轻仓参与。",
            key_risks=["短期涨幅过大回调风险", "行业价格战", "估值偏高"],
        ),
        "601318": ResearchVerdict(
            code="601318", name="中国平安", direction="hold",
            confidence=0.55, target_price=50.00, risk_level="low",
            core_reasoning="估值处于历史底部具有安全边际，基本面逐步改善。"
                           "但技术面方向不明，资金面中性，短期缺乏催化剂。建议观望等待信号。",
            key_risks=["利率下行影响利差", "保险需求复苏不及预期"],
        ),
        "000333": ResearchVerdict(
            code="000333", name="美的集团", direction="hold",
            confidence=0.60, target_price=68.00, risk_level="low",
            core_reasoning="基本面稳健，海外业务增长强劲，估值合理偏低。"
                           "但技术面尚未完全确认上行趋势，资金面中性。可关注但不急于买入。",
            key_risks=["海外关税不确定性", "国内消费复苏节奏"],
        ),
    }


# ── 风控约束 ────────────────────────────────────

def _make_position_limits() -> dict[str, PositionLimit]:
    """生成风控仓位上限"""
    limits = {}
    configs = [
        ("600519", "贵州茅台", 0.20, 500, 840250, 1.8, ["无"]),
        ("000858", "五粮液",  0.15, 400, 60920,  2.1, ["无"]),
        ("300750", "宁德时代", 0.12, 200, 41120,  3.2, ["高波动率"]),
        ("002594", "比亚迪",   0.10, 100, 26800,  4.5, ["高波动率", "短期涨幅过大"]),
        ("601318", "中国平安",  0.15, 1200, 57840, 1.2, ["无"]),
        ("000333", "美的集团",  0.15, 900,  56520, 1.5, ["无"]),
    ]
    for code, name, pct, shares, value, vol, flags in configs:
        limits[code] = PositionLimit(
            code=code, name=name,
            max_position_pct=pct,
            max_shares=shares,
            max_value=value,
            volatility=vol,
            risk_flags=flags,
        )
    return limits


# ── 组合决策 ────────────────────────────────────

def _make_portfolio_result() -> PortfolioResult:
    """生成最终组合决策: 买入 600519 + 000858"""
    decisions = [
        FinalDecision(symbol="600519", symbol_name="贵州茅台", volume=200, entry_price=1680.50),
        FinalDecision(symbol="000858", symbol_name="五粮液",   volume=300, entry_price=152.30),
    ]
    # 200 * 1680.50 + 300 * 152.30 = 336100 + 45690 = 381790
    cash_used = 200 * 1680.50 + 300 * 152.30
    return PortfolioResult(
        decisions=decisions,
        cash_used=cash_used,
        cash_remaining=500_000.0 - cash_used,
        total_positions=2,
        risk_summary="整体风险可控。茅台+五粮液同属白酒板块，行业集中度需关注。"
                     "建议总仓位不超过40%，剩余资金保留灵活性。",
    )
