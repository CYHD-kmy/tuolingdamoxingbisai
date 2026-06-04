# 智投未来 — A股日内投资智能体

基于 LangGraph 多智能体协作的 A 股日内投资决策系统。每日自动完成「全市场扫描→多因子筛选→四维分析师并行分析→多空辩论→风控约束→组合决策→JSON 输出」全链路，输出可解释、可审计的投资建议。

详见 [产品设计书](产品设计书.md)。

## 设计原则

| 原则 | 说明 |
|------|------|
| **博采众长** | 借鉴 TradingAgents-CN(辩论/LLM适配)、ai-hedge-fund(风控思路)、daily_stock_analysis(因子筛选) 的成熟模式，自研 A 股三层仓位框架等 |
| **可解释性** | 每笔决策附带完整推理链，数据→信号→决策全链路可审计 |
| **稳健性** | 多层数据源降级、异常熔断、空仓兜底，极端行情下不崩溃 |
| **模块化** | 数据层 / 分析层 / 决策层 / 输出层 松耦合，各层可独立替换 |

## 架构

```
调度层 (Scheduler / CLI)
  │
  ├─ 数据层 (Data Layer)
  │   Tushare → BaoStock → AKShare  三层降级 (Tushare 为主力，AKShare 为兜底)
  │   UnifiedDataInterface  统一数据接口 + 缓存
  │
  ├─ 分析层 (Analysis Layer)
  │   阶段一: 海选筛选  5000+ → 10因子打分 → Top 20
  │   阶段二: 深度分析  四维分析师 (技术/基本/资金/消息) × quick LLM
  │   阶段三: 辩论对抗  多头 ↔ 空头 (max 3轮) → 研究主管 (deep LLM)
  │
  ├─ 决策层 (Decision Layer)
  │   风控主管 (确定性规则) → 投资组合主管 (deep LLM) → 最终买入决策
  │
  └─ 输出层 (Output Layer)
      赛道 JSON + 推理追踪日志 (+ 可选的 Markdown 日报)
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LLM_API_KEY=sk-xxx

# 3. 运行 (演示模式，无需网络和 API)
python -m src.main --demo

# 4. 运行 (正常模式，需要 API Key)
python -m src.main
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key (必填) | - |
| `LLM_QUICK_MODEL` | quick 模型 (分析师用) | `deepseek-chat` |
| `LLM_DEEP_MODEL` | deep 模型 (决策主管用) | `deepseek-reasoner` |
| `LLM_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `TUSHARE_TOKEN` | Tushare Token (可选) | - |

## 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 数据层 | `src/data/` | 多源降级编排 (Tushare → BaoStock → AKShare) + 缓存 + 数据质量标记 |
| 海选筛选 | `src/screening/` | ST/停牌/新股过滤 + 10因子加权打分 + ETF筛选 |
| 分析Agent | `src/agents/` | 技术面/基本面/资金面/消息面 四维分析 + ETF分析师 + 多空辩论 + 管理团队 |
| 工作流 | `src/graph/` | LangGraph 状态管理 + 流水线编排 |
| LLM 适配 | `src/llm/` | OpenAI-compatible 客户端 (DeepSeek/OpenAI)，quick/deep 分层 |
| 输出层 | `src/output/` | JSON 格式化 + 约束校验 + 追踪日志 |
| 降级策略 | `src/agents/fallback.py` | LLM 不可用时确定性规则引擎接管全链路 |
| 工具 | `src/utils/` | 配置管理 / 交易日历 / 输出校验 |

## 海选筛选 — 10 因子打分

| 因子类别 | 因子名称 | 权重 | 说明 |
|----------|----------|------|------|
| 趋势 | 均线多头排列 | 12% | MA5 > MA10 > MA20 > MA60 |
| 动量 | 5日涨幅 | 10% | 正向动量 |
| 量价 | 放量上涨 | 12% | 量比>1.5 加分 + 涨幅连续评分 |
| 资金 | 主力净流入 | 15% | 大单净流入占比 |
| 北向 | 外资持仓变化 | 10% | 北向持股占比趋势 |
| 情绪 | 交易活跃度 | 8% | 换手率+连续阳线 |
| 质量 | PE+ROE+营收增速+经营现金流 | 10% | 多维度基本面过滤 |
| 风险 | 波动率适中 | 8% | 排除异常波动股 |
| 流动性 | 日均成交额 | 10% | > 5000 万，确保可交易 |
| 筹码 | 股东人数变化 | 5% | 筹码集中度指标 |

筛选流程: `5000+ → 剔除 ST/*ST/停牌/新股(上市<60天) → 剔除 日均成交额<5000万 → 10因子加权打分 → Top 20`

### 四维分析师并行分析

Top 20 候选池中每只股票由 4 个分析师**并行**分析（各自使用 quick LLM）：

| 分析师 | 分析维度 | 核心指标 |
|--------|----------|----------|
| 技术面 | 趋势形态、超买超卖、支撑压力 | 均线排列、MACD、RSI、布林带、筹码分布 |
| 基本面 | 估值水平、盈利质量、成长性 | PE/PB 分位数、ROE 趋势、营收增速、公告影响 |
| 资金面 | 主力动向、北向资金、杠杆变化 | 大单净流入、融资融券余额、龙虎榜 |
| 消息面 | 新闻情感、热点匹配、市场热度 | 财经新闻正负面、行业政策、社交媒体热度 |

每个分析师支持最多 3 轮 Tool-Calling 自主调用数据。

## 多空辩论流程

```
多头研究员: "基于技术面金叉+资金流入+业绩预增，建议买入..."
        ↓
空头研究员: 逐条反驳多头论点
        ↓
多头研究员: 回击空头论点
        ↓ (max 3 轮)
研究主管 (deep LLM): 综合双方论点 → 最终研报
  {方向, 置信度, 目标价, 风险等级, 核心理由}
```

## 风控参数

### 三层仓位框架

| 层级 | 单票上限 | 总仓位上限 | 说明 |
|------|----------|------------|------|
| 核心仓 (Core) | ≤ 40% | ≤ 40% (≤2只) | 高置信度 + 低PE + 大市值标的 |
| 卫星仓 (Satellite) | ≤ 14% | ≤ 35% (≤4只) | 其他合格候选 |
| 现金备用 (Cash) | — | ≥ 25% | 始终保留现金缓冲 |

### 其他风控规则

| 规则 | 值 | 说明 |
|------|-----|------|
| 行业集中度 | ≤ 40% | 同行业总仓位 |
| 日换手率 | ≤ 50% | 避免过度交易 |
| 日内熔断 | 5% | 回撤超 5% 暂停交易 |
| 相关性惩罚 | ×0.7 | 与已持仓高相关 (ρ>0.7) 降低仓位 |
| 波动率调整 | 低波×1.25 / 中波×1.0 / 高波×0.5 | 波动率自适应仓位 |
| 开盘大跌过滤 | ≤ -5% | 单日开盘跌幅 > 5% 剔除个体 |
| 全面下跌熔断 | ≥ 3000 只下跌 | 全市场 > 3000 只下跌强制空仓 |
| 限售解禁 | 解禁 > 5% ×0.3, > 2% ×0.6 | 近期限售解禁减仓 |

## LLM 分层策略

| 层级 | 模型 | 用途 | 调用频率 |
|------|------|------|----------|
| quick | DeepSeek-V3.2 / GPT-4o-mini | 四维分析师、多空研究员 | 高 (每只股票 × 4) |
| deep | DeepSeek-V4-Pro / Claude Opus | 研究主管、组合主管 | 低 (全市场 × 1) |

## 策略可解释性

每笔决策可从最终输出回溯到原始数据：

```
最终决策: 买入贵州茅台 200 股
  ↑ 组合主管: "综合看多信号，风控允许500股，保守配置200股"
  ↑ 研究主管: "多头(业绩确定性+资金流入)强于空头(估值偏高+短期超买)"
  ↑ ├─ 多头研究员: "Q1超预期20%, 北向连续5日净流入, 60日均线支撑有效"
    ├─ 空头研究员: "PE处于70%分位, RSI接近超买, 短期回调风险"
    ├─ 技术面: bullish(0.75)  ├─ 基本面: bullish(0.82)
    ├─ 资金面: bullish(0.68)  └─ 消息面: bullish(0.70)
```

每交易日保留完整 JSON 轨迹文件，支持按日期回溯审计。

## 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 数据源接口不稳定 | 数据缺失 | 三层降级 (Tushare→BaoStock→AKShare) + 缓存兜底 |
| LLM 调用成本过高 | 预算超支 | 海选筛选减少调用量 + quick/deep 分层 |
| 市场极端波动 | 大幅回撤 | 日内5%熔断 + 单票仓位上限 + 保留现金 |
| 过拟合历史数据 | 实盘表现差 | 辩论机制引入对抗观点 + 多分析师交叉验证 |
| 平台对接问题 | 无法提交 | 输出格式严格校验 (volume取整/预算/风控) |

## 异常处理

| 场景 | 处理策略 |
|------|----------|
| 数据源全部不可用 | 输出空数组 `[]`，标记"数据不可用" |
| LLM 调用超时 | 降级为确定性规则引擎 (`fallback.py`): 分析师→规则指标、研究主管→投票、组合主管→保守配置 |
| 单只股票分析失败 | 跳过该股票，继续分析其他候选 |
| 组合构建失败 | 输出空数组 `[]` |
| 当日无符合条件标的 | 输出空数组 `[]` |

## 输出格式

### 赛道 JSON

```json
[
  {"symbol": "600519", "symbol_name": "贵州茅台", "volume": 200},
  {"symbol": "000858", "symbol_name": "五粮液", "volume": 500}
]
```

自动校验：volume 取整到 100 的倍数、总金额不超可用现金、单票不超风控上限、停牌股自动剔除。

### 推理追踪

每笔决策保留完整 JSON 轨迹（筛选评分→四维报告→辩论记录→风控计算→最终决策），支持全链路回溯审计。

## 运行方式

```bash
# 正常模式 (需要 LLM API Key 和数据源)
python -m src.main

# 演示模式 (使用样本数据，无需网络和 API)
python -m src.main --demo

# 或在 Python 代码中调用
from src.graph.workflow import run_pipeline
state = run_pipeline(total_capital=500_000.0, available_cash=500_000.0)
```

## 目录结构

```
zhitou-future/
├── README.md
├── 产品设计书.md
├── requirements.txt
├── .env.example
├── manage.sh                         # 服务管理脚本
├── manage.py                         # 跨平台管理脚本
│
├── src/
│   ├── main.py                       # 主入口
│   ├── scheduler.py                  # 定时调度
│   ├── demo.py                       # 演示数据
│   │
│   ├── data/                         # 数据层
│   │   ├── interface.py              #   统一数据接口
│   │   ├── cache.py                  #   TTL 缓存 + 磁盘持久化
│   │   └── fetchers/                 #   数据源适配器
│   │       ├── akshare_fetcher.py    #     AKShare (兜底)
│   │       ├── tushare_fetcher.py    #     Tushare (主力)
│   │       └── baostock_fetcher.py   #     BaoStock (备用)
│   │
│   ├── agents/                       # Agent 层
│   │   ├── base.py                   #   Agent 基类
│   │   ├── models.py                 #   数据模型
│   │   ├── tools.py                  #   Agent 工具集
│   │   ├── fallback.py              #   LLM 降级: 确定性规则引擎
│   │   ├── analysts/                 #   四维分析师
│   │   │   ├── technical.py          #     技术面
│   │   │   ├── fundamentals.py       #     基本面
│   │   │   ├── fund_flow.py          #     资金面
│   │   │   ├── news_sentiment.py     #     消息面
│   │   │   └── etf.py                #     ETF 分析师
│   │   ├── researchers/              #   辩论研究员
│   │   │   ├── bull.py               #     多头研究员
│   │   │   ├── bear.py               #     空头研究员
│   │   │   └── engine.py             #     辩论引擎
│   │   └── managers/                 #   决策主管
│   │       ├── research_manager.py   #     研究主管
│   │       ├── risk_manager.py       #     风控主管
│   │       └── portfolio_manager.py  #     组合主管
│   │
│   ├── graph/                        # LangGraph 编排
│   │   ├── state.py                  #   共享状态定义
│   │   └── workflow.py               #   流水线构建
│   │
│   ├── llm/                          # LLM 适配层
│   │   ├── factory.py                #   模型工厂
│   │   ├── client.py                 #   OpenAI-compatible 客户端
│   │   └── schema.py                 #   LLM 输出 Schema
│   │
│   ├── screening/                    # 海选筛选
│   │   ├── pipeline.py               #   筛选管道
│   │   ├── scorer.py                 #   10因子打分
│   │   ├── filters.py                #   过滤
│   │   └── etf_screener.py           #   ETF 筛选器
│   │
│   ├── output/                       # 输出层
│   │   ├── json_formatter.py         #   赛道 JSON 格式 + 校验
│   │   ├── report_generator.py       #   Markdown 日报
│   │   └── trace_logger.py           #   推理追踪日志
│   │
│   └── utils/                        # 工具
│       ├── config.py                 #   配置管理
│       ├── trading_calendar.py       #   A股交易日历
│       └── validators.py             #   输出校验
│
├── results/                          # 运行结果 (自动生成)
│   ├── trace_YYYYMMDD.json           #   推理轨迹
│   └── report_YYYYMMDD.md            #   日报
│
└── tests/                            # 测试 (8 个文件)
    ├── conftest.py
    ├── test_agents.py
    ├── test_data.py
    ├── test_integration.py
    ├── test_llm.py
    ├── test_output.py
    ├── test_screening.py
    └── test_trading_calendar.py
```
