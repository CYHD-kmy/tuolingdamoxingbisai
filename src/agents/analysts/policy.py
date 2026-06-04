"""
政策面分析师 — A 股政策市分析。

A 股是典型的政策驱动市场:
- 监管政策 (证监会/交易所新规)
- 产业政策 (国家扶持/限制目录)
- 窗口指导
- 重大宏观政策 (降息降准/财政刺激)

政策影响通常持续数周到数月，是中线选股的核心因子。
"""

from __future__ import annotations

from ..base import BaseAnalyst
from ..tools import tools_for

POLICY_PROMPT = """你是一位资深的 A 股政策面分析师，专精于解读宏观政策和产业政策对股市的影响。

## 分析框架

### 1. 产业政策判定
- 该股票所在行业是否受国家产业政策扶持? (如新能源/半导体/人工智能等)
- 是否属于被限制或调控的行业? (如地产/教培等)
- 近期是否有重大产业政策发布? (国务院/发改委/工信部文件)

### 2. 监管环境评估
- 近期是否有与该公司/行业相关的监管新规?
- 监管态势是趋严还是放松?
- 公司是否面临处罚/调查/整改等监管风险?

### 3. 宏观政策传导
- 货币政策 (降息/降准/流动性投放) 对该行业的影响
- 财政政策 (减税/补贴/基建投资) 的受益程度
- 汇率/贸易政策对出口型企业的影响

### 4. 政策持续性判断
- 政策是短期刺激还是长期战略?
- 政策执行到业绩兑现的传导周期有多长?

## 输出格式 (JSON)
```json
{
    "analyst_type": "policy",
    "code": "<股票代码>",
    "name": "<股票名称>",
    "signal": "bullish / bearish / neutral",
    "confidence": 0.0-1.0,
    "reasoning": "核心推理 (200字以内)",
    "key_points": ["关键发现1", "关键发现2", ...],
    "risks": ["政策风险1", "政策风险2", ...]
}
```

## 注意事项
- A 股政策影响权重极大，牛市/熊市转换常由政策触发
- 产业政策落地到业绩兑现通常需要 3-12 个月
- 注意区分"政策利好"和"政策已经充分定价"的股票
- 政策突变 (如行业整顿) 可能导致估值崩塌，需高度警惕
"""


class PolicyAnalyst(BaseAnalyst):
    """政策面分析师"""

    analyst_type = "policy"

    @property
    def system_prompt(self) -> str:
        return POLICY_PROMPT

    @property
    def tools(self) -> list:
        return tools_for("policy")

    def build_context(self, code: str) -> str:
        stock_info = self._data.get_stock_info(code)
        industry = stock_info.get("industry", "未知行业")
        name = stock_info.get("name", "")

        # 获取近期新闻 (政策相关)
        from datetime import datetime
        news = []
        try:
            policy_news = self._data.get_news(f"{name or code} 政策", days=30)
            news.extend(policy_news[:5])
            industry_news = self._data.get_news(f"{industry} 政策", days=30)
            news.extend(industry_news[:5])
        except Exception:
            pass

        # 获取公告
        announcements = self._data.get_announcements(code, days=30)

        lines = [
            f"股票: {name or code} ({code})",
            f"所属行业: {industry}",
        ]

        if news:
            lines.append(f"\n近期政策相关新闻 ({len(news)} 条):")
            for i, n in enumerate(news[:8]):
                title = n.get("title", n.get("content", ""))[:100]
                source = n.get("source", "")
                date = n.get("date", n.get("time", ""))
                lines.append(f"  {i+1}. [{date}] {title} ({source})")

        if announcements:
            lines.append(f"\n近期公告 ({len(announcements)} 条):")
            for i, a in enumerate(announcements[:5]):
                title = a.get("title", a.get("content", ""))[:120]
                date = a.get("date", a.get("time", ""))
                lines.append(f"  {i+1}. [{date}] {title}")

        lines.append(f"\n请根据上述信息，分析 {name or code} ({industry}) 的政策面影响。")
        return "\n".join(lines)
