# 智投未来 — A股日内投资智能体

基于 LangGraph 多智能体协作的 A 股日内投资决策系统。每日自动完成「全市场扫描→多因子筛选→四维分析师并行分析→多空辩论→风控约束→组合决策→JSON 输出」全链路，输出可解释、可审计的投资建议。

详见 [产品设计书](产品设计书.md)。

## 设计原则

| 原则 | 说明 |
|------|------|
| **博采众长** | 最大化复用 TradingAgents-CN、ai-hedge-fund、daily_stock_analysis 的成熟模块 |
| **可解释性** | 每笔决策附带完整推理链，数据→信号→决策全链路可审计 |
| **稳健性** | 多层数据源降级、异常熔断、空仓兜底，极端行情下不崩溃 |
| **模块化** | 数据层 / 分析层 / 决策层 / 输出层 松耦合，各层可独立替换 |

## 架构

```
调度层 (Scheduler / CLI)
  │
  ├─ 数据层 (Data Layer)
  │   AKShare → Tushare → BaoStock  三级降级 (Tushare Token可用时自动提升为优先)
  │   UnifiedDataInterface  统一数据接口 + 缓存
  │
  ├─ 分析层 (Analysis Layer)
  │   阶段一: 海选筛选  5000+ → 10因子打分 → Top 20
  │   阶段二: 深度分析  四维分析师 (技术/基本/资金/消息) × quick LLM
  │   阶段三: 辩论对抗  多头 ↔ 空头 (max 3轮) → 研究主管 (deep LLM)
  │
  ├─ 决策层 (Decision Layer)
  │   风控主管 (确定性规则) → 投资组合主管 (deep LLM) → 最终决策
  │
  ├─ 输出层 (Output Layer)
  │   赛道 JSON + Markdown 日报 + 推理追踪日志
  │
  └─ 展示层 (Web Dashboard)
      FastAPI 看板 — 首页 / 看板 / 历史 / 日报
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

# 4. 运行 (正常模式)
python -m src.main

# 5. 高级功能
python -m src.main --backtest 20260501 20260526                  # 历史回测
python -m src.main --benchmark 000300                             # 回测基准指数 (可选，默认沪深300)
python -m src.main --strategy momentum,quality                    # 多策略竞争 (5种: momentum/mean_reversion/quality/sentiment/all)
python -m src.main --rl-train --rl-episodes 200                   # 强化学习训练
python -m src.main --rl-model results/rl_model.json               # 加载RL模型推断

# 6. Transformer 时序编码器 (Phase 4)
python -m src.main --transformer-train --transformer-epochs 100   # 训练 Transformer
export TRANSFORMER_ENABLED=true                                    # 启用 Transformer
python -m src.main --demo                                          # 推理 (融合Transformer评分)
```

### 启动 Web 看板

```bash
# 方式一: 跨平台管理脚本 (推荐)
python manage.py start       # 启动服务
python manage.py status      # 查看状态
python manage.py stop        # 停止服务

# 方式二: Shell 管理脚本 (macOS/Linux)
./manage.sh start       # 启动服务
./manage.sh status      # 查看状态
./manage.sh stop        # 停止服务

# 方式三: 直接启动
python -m src.api.server

# 访问: http://localhost:8000
#   /home        首页
#   /dashboard   实时看板
#   /history     历史记录
#   /report      日报
```

### 开机自启

```bash
# macOS (launchd)
./manage.sh install     # 安装 launchd 服务
./manage.sh uninstall   # 卸载

# Linux (systemd)
python manage.py install     # 安装 systemd 用户服务
python manage.py uninstall   # 卸载

# Windows
python manage.py install     # 显示任务计划程序配置指引
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_API_KEY` | LLM API Key (必填) | - |
| `LLM_QUICK_MODEL` | quick 模型 (分析师用) | `deepseek-chat` |
| `LLM_DEEP_MODEL` | deep 模型 (决策主管用) | `deepseek-reasoner` |
| `LLM_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `TUSHARE_TOKEN` | Tushare Token (可选) | - |
| `ZHITOU_HOST` | 看板监听地址 | `0.0.0.0` |
| `ZHITOU_PORT` | 看板端口 | `8000` |

## 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 数据层 | `src/data/` | 多源降级编排 (AKShare → Tushare → BaoStock) + 缓存 + 数据质量标记 (live/cached/fallback/stale) |
| 海选筛选 | `src/screening/` | ST/停牌/新股过滤 + 10因子加权打分 + ETF筛选 (已接入主流水线，支持股/ETF并行+混合组合) |
| 分析Agent | `src/agents/` | 技术面/基本面/资金面/消息面 四维分析 + ETF分析师 + 多空辩论 + 管理团队 |
| 工作流 | `src/graph/` | LangGraph 状态管理 + 流水线编排 |
| LLM 适配 | `src/llm/` | OpenAI-compatible 客户端 (DeepSeek/OpenAI)，quick/deep 分层 |
| 输出层 | `src/output/` | JSON 格式化 + 约束校验 + Markdown 日报 + 追踪日志 |
| Web 看板 | `src/api/` | FastAPI 多页面看板 (首页/看板/历史/日报) + 9个 API 端点 |
| 降级策略 | `src/agents/fallback.py` | LLM 不可用时确定性规则引擎接管全链路 |
| 持仓追踪 | `src/agents/portfolio_tracker.py` | 跨交易日持仓管理: 成本基价/浮动盈亏/行业暴露/日收益历史 |
| 盘中监控 | `src/monitoring/` | 30s轮询: 止损(-7%)/止盈(+15%)/熔断(-5%)/Webhook告警 |
| 记忆系统 | `src/memory/` | ChromaDB 向量存储: 历史行情索引/相似检索/置信度校准 (需手动取消 requirements.txt 中 chromadb 注释以安装) |
| 回测框架 | `src/backtesting/` | 历史回放: Sharpe/MaxDD/Calmar/ProfitFactor + JSON+MD报告 |
| 组合优化 | `src/optimization/` | 风险平价(ERC)/最小方差/最大分散化 三种权重分配方法 |
| 多策略 | `src/strategies/` | 5种Alpha策略并行竞争: 动量/均值回归/质量/情绪/默认10因子 (CompetitionEngine.save_performance 已实现，待接入主流程自动调用) |
| 强化学习 | `src/rl/` | DQN智能体: 手动神经网络 + 经验回放 + 交易环境, 无PyTorch依赖 |
| Transformer | `src/transformer/` | 轻量 Transformer 时序编码器: 2层/4头/32维, 纯Python实现, 增强评分+RL特征 |
| 工具 | `src/utils/` | 配置管理 / 交易日历 / 输出校验 |

## 海选筛选 — 10 因子打分

| 因子类别 | 因子名称 | 权重 | 说明 |
|----------|----------|------|------|
| 趋势 | 均线多头排列 | 12% | MA5 > MA10 > MA20 > MA60 |
| 动量 | 5日涨幅 | 10% | 正向动量 |
| 量价 | 放量上涨 | 12% | 量比 > 1.5 且涨幅 > 2% |
| 资金 | 主力净流入 | 15% | 大单净流入占比 |
| 北向 | 外资持仓变化 | 10% | 北向持股占比趋势 |
| 情绪 | 交易活跃度 | 8% | 换手率+连续阳线 (MVP替代新闻情感) |
| 质量 | PE合理性+ROE/毛利率 | 10% | 多维度基本面过滤 |
| 风险 | 波动率适中 | 8% | 排除异常波动股 |
| 流动性 | 日均成交额 | 10% | > 5000 万，确保可交易 |
| 筹码 | 股东人数变化 | 5% | 筹码集中度指标 |

筛选流程: `5000+ → 剔除 ST/*ST/停牌/新股(上市<60天) → 剔除 日均成交额<5000万 (流动性过滤) → 10因子加权打分 → Top 20`

### 四维分析师并行分析

Top 20 候选池中每只股票由 4 个分析师**并行**分析（各自使用 quick LLM）：

| 分析师 | 分析维度 | 核心指标 |
|--------|----------|----------|
| 技术面 | 趋势形态、超买超卖、支撑压力 | 均线排列、MACD、RSI、布林带、筹码分布 |
| 基本面 | 估值水平、盈利质量、成长性 | PE/PB 分位数、ROE 趋势、营收增速、公告影响 |
| 资金面 | 主力动向、北向资金、杠杆变化 | 大单净流入、融资融券余额、龙虎榜 |
| 消息面 | 新闻情感、热点匹配、市场热度 | 财经新闻正负面、行业政策、社交媒体热度 |

每个分析师支持最多 3 轮 Tool-Calling 自主调用数据，防止无限循环。

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

| 规则 | 值 | 说明 |
|------|-----|------|
| 单票仓位上限 | ≤ 20% | 波动率动态调整 (低波×1.25, 高波×0.50) |
| 行业集中度 | ≤ 40% | 同行业总仓位 |
| 日换手率 | ≤ 50% | 避免过度交易 |
| 日内熔断 | 5% | 回撤超 5% 暂停交易 |
| 现金保留 | ≥ 10% | 始终保留现金缓冲 |
| 相关性惩罚 | ×0.7 | 与已持仓高相关 (ρ>0.7) 降低仓位 |

## LLM 分层策略

| 层级 | 模型 | 用途 | 调用频率 |
|------|------|------|----------|
| quick | DeepSeek-V3.2 / GPT-4o-mini | 四维分析师、多空研究员 | 高 (每只股票 × 4) |
| deep | DeepSeek-V4-Pro / Claude Opus | 研究主管、组合主管 | 低 (全市场 × 1) |

## 技术选型与复用

| 模块 | 来源项目 | 复用程度 | 适配工作 |
|------|----------|----------|----------|
| 数据源适配器 | daily_stock_analysis `data_provider/` | 90% | 精简为 A 股核心数据源 |
| 多因子筛选 | daily_stock_analysis `stock_analyzer.py` | 80% | 提取多因子打分逻辑 |
| Agent 框架 | TradingAgents-CN `agents/` + `graph/` | 70% | 简化为4分析+2研究员+1主管 |
| 辩论模式 | TradingAgents-CN Bull/Bear | 80% | 调整辩论轮数和提示词 |
| LLM 适配层 | TradingAgents-CN `llm_clients/` | 95% | 直接复用 |
| 风控模块 | ai-hedge-fund `risk_manager.py` | 60% | 从美股适配到A股规则 |
| 组合管理 | ai-hedge-fund `portfolio_manager.py` | 50% | 修改输出格式为赛道JSON |

技术栈: LangGraph (编排) / AKShare + Tushare + BaoStock (数据) / FastAPI (Web) / ChromaDB (记忆) / APScheduler (调度)

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
| AKShare 接口不稳定 | 数据缺失 | 三层降级 (AKShare→Tushare→BaoStock) + 缓存兜底 |
| LLM 调用成本过高 | 预算超支 | 海选筛选减少调用量 + quick/deep 分层 |
| 市场极端波动 | 大幅回撤 | 日内5%熔断 + 单票仓位上限 + 保留现金 |
| 过拟合历史数据 | 实盘表现差 | 辩论机制引入对抗观点 + 多分析师交叉验证 |
| 平台对接问题 | 无法提交 | 输出格式严格校验 (volume取整/预算/风控) |

## 交易日时间线

```
08:30  系统唤醒，检查数据源可用性
09:00  阶段一: 全市场海选 → Top 20 候选池
09:15  阶段二: 四维分析师并行分析
10:00  阶段三: 辩论式对抗 + 研究主管研判
10:30  阶段四: 风控计算 + 组合构建
10:45  输出赛道 JSON 格式决策 → 提交至擂台平台
11:00  盘中监控启动 (可选: 熔断/止盈止损信号)
14:30  最终决策确认并输出
15:00  收盘后: 生成日报 + 更新记忆库 + 保存推理轨迹
```

## 异常处理

| 场景 | 处理策略 |
|------|----------|
| 数据源全部不可用 | 输出空数组 `[]`，标记"数据不可用" |
| LLM 调用超时 | 降级为确定性规则引擎 (`fallback.py`): 分析师→规则指标、研究主管→投票、组合主管→保守配置 |
| 单只股票分析失败 | 跳过该股票，继续分析其他候选 |
| 组合构建失败 | 输出空数组 `[]`，保留现有持仓 |
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

## 迭代路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| **第一阶段 (MVP)** | 数据管道 + 四分析师 + 辩论 + 风控 + JSON 输出 + Web 看板 | 已完成 |
| **第二阶段 (增强)** | ChromaDB 记忆系统、盘中实时监控、ETF 策略、持仓追踪 | 已完成 |
| **第三阶段 (进化)** | 回测框架、风险平价优化、多策略竞争引擎、DQN 强化学习 | 已完成 |
| **第四阶段 (前沿)** | 轻量 Transformer 时序编码器 ✅、实盘接入 (规划中)、多资产扩展 (规划中) | 部分完成 |

## 目录结构

```
zhitou-future/
├── README.md
├── 产品设计书.md
├── requirements.txt
├── .env.example
├── manage.sh                         # 服务管理脚本 (macOS/Linux)
├── manage.py                         # 跨平台管理脚本 (Windows/macOS/Linux)
│
├── src/
│   ├── main.py                       # 主入口
│   ├── scheduler.py                  # 定时调度
│   ├── demo.py                       # 演示数据
│   │
│   ├── data/                         # 数据层
│   │   ├── interface.py              #   统一数据接口
│   │   ├── cache.py                  #   数据缓存
│   │   └── fetchers/                 #   数据源适配器
│   │       ├── akshare_fetcher.py    #     AKShare (主力，免费)
│   │       ├── tushare_fetcher.py    #     Tushare (增强，需Token)
│   │       └── baostock_fetcher.py   #     BaoStock (兜底，免费)
│   │
│   ├── agents/                       # Agent 层
│   │   ├── base.py                   #   Agent 基类
│   │   ├── models.py                 #   数据模型
│   │   ├── tools.py                  #   Agent 工具集 (Tool-Calling 3轮上限)
│   │   ├── analysts/                 #   四维分析师
│   │   │   ├── technical.py          #     技术面 (均线/MACD/RSI/布林带)
│   │   │   ├── fundamentals.py       #     基本面 (PE/PB/ROE/营收增速)
│   │   │   ├── fund_flow.py          #     资金面 (主力/北向/融资融券)
│   │   │   ├── news_sentiment.py     #     消息面 (新闻情感/热点匹配)
│   │   │   └── etf.py                #     ETF 分析师
│   │   ├── researchers/              #   辩论研究员
│   │   │   ├── bull.py               #     多头研究员
│   │   │   ├── bear.py               #     空头研究员
│   │   │   └── engine.py             #     辩论引擎 (max 3轮 + 收敛检测)
│   │   ├── managers/                 #   决策主管
│   │   │   ├── research_manager.py   #     研究主管 (deep LLM 综合研判)
│   │   │   ├── risk_manager.py       #     风控主管 (确定性规则)
│   │   │   └── portfolio_manager.py  #     组合主管 (deep LLM 最终配置)
│   │   ├── fallback.py              #   LLM 降级: 确定性规则引擎 (无LLM时全链路接管)
│   │   └── portfolio_tracker.py     #   跨日持仓追踪 (成本/盈亏/行业暴露/日收益历史)
│   │
│   ├── graph/                        # LangGraph 编排
│   │   ├── state.py                  #   共享状态定义
│   │   └── workflow.py               #   流水线构建
│   │
│   ├── llm/                          # LLM 适配层
│   │   ├── factory.py                #   模型工厂
│   │   ├── client.py                 #   OpenAI-compatible 客户端 (全局并发限流)
│   │   └── schema.py                 #   LLM 输出 Schema
│   │
│   ├── screening/                    # 海选筛选
│   │   ├── pipeline.py               #   筛选管道
│   │   ├── scorer.py                 #   10因子打分
│   │   ├── filters.py                #   ST/停牌/新股/流动性过滤
│   │   └── etf_screener.py           #   ETF 筛选器
│   │
│   ├── backtesting/                  # 回测框架
│   │   ├── engine.py                 #   逐日历史回放引擎
│   │   ├── metrics.py                #   绩效指标 (Sharpe/MaxDD/Calmar/ProfitFactor/WinRate)
│   │   └── report.py                 #   JSON+MD 双格式报告
│   │
│   ├── optimization/                 # 组合优化
│   │   └── risk_parity.py            #   ERC / MinVariance / MaxDiversification 权重分配
│   │
│   ├── strategies/                   # 多策略竞争
│   │   ├── engine.py                 #   并行竞争引擎 + 软投票合并
│   │   ├── base.py                   #   策略基类
│   │   ├── registry.py               #   策略注册表
│   │   ├── momentum.py               #   趋势动量策略
│   │   ├── mean_reversion.py         #   均值回归策略
│   │   ├── quality.py                #   质量价值策略
│   │   ├── sentiment.py              #   情绪资金策略
│   │   └── default_strategy.py       #   默认10因子策略
│   │
│   ├── rl/                           # 强化学习
│   │   ├── agent.py                  #   DQN智能体 + SimpleNN (手动BP)
│   │   ├── environment.py            #   Gym风格交易环境
│   │   ├── features.py               #   7维技术特征提取
│   │   └── trainer.py                #   跨股票训练器
│   │
│   ├── transformer/                   # Transformer 时序编码器 (Phase 4)
│   │   ├── __init__.py                #   公开 API
│   │   ├── features.py                #   StockDaily → 10维特征向量 + Z-score 归一化
│   │   ├── embedding.py               #   输入投影 (Linear 10→32) + 正弦位置编码
│   │   ├── attention.py               #   多头自注意力 (4头) + Softmax + 矩阵运算
│   │   ├── encoder.py                 #   TransformerEncoderLayer + TransformerEncoder (2层)
│   │   ├── model.py                   #   StockTransformer 完整模型 + JSON 序列化
│   │   ├── training.py                #   训练数据生成(滑动窗口) + 训练循环(SGD + 解析反向传播)
│   │   └── scorer.py                  #   TransformerScorer (与手工因子融合)
│   │
│   ├── monitoring/                   # 盘中监控
│   │   └── monitor.py                #   30s轮询: 止损(-7%)/止盈(+15%)/熔断(-5%) + Webhook告警
│   │
│   ├── memory/                       # 向量记忆
│   │   └── __init__.py              #   ChromaDB 历史行情索引
│   │
│   ├── api/                          # Web 看板
│   │   ├── server.py                 #   FastAPI 服务 (9 端点 + 5 页面路由)
│   │   └── static/                   #   前端静态文件 (首页/看板/历史/日报)
│   │
│   ├── output/                       # 输出层
│   │   ├── json_formatter.py         #   赛道 JSON 格式 + 硬约束校验 (volume取整/预算/风控)
│   │   ├── report_generator.py       #   Markdown 日报 (含持仓快照/推理链/明日关注)
│   │   └── trace_logger.py           #   推理轨迹 (全链路回溯审计)
│   │
│   └── utils/                        # 工具
│       ├── config.py                 #   配置管理 (环境变量 + .env)
│       ├── trading_calendar.py       #   A股交易日历
│       └── validators.py             #   输出校验
│
├── results/                          # 运行结果 (自动生成)
│   ├── trace_YYYYMMDD.json           #   推理轨迹
│   └── report_YYYYMMDD.md            #   日报
│
└── tests/                            # 测试 (153 tests, 13 个文件)
    ├── conftest.py                   #   pytest 配置
    ├── test_agents.py                #   Agent 层 (风控/辩论/模型/JSON解析)
    ├── test_api.py                   #   FastAPI 端点
    ├── test_backtesting.py           #   回测 (Sharpe/MaxDD/引擎/报告)
    ├── test_data.py                  #   数据层 (缓存/模型/代码标准化)
    ├── test_integration.py           #   端到端 (demo全链路 + 校验)
    ├── test_llm.py                   #   LLM 客户端/Schema/工厂
    ├── test_optimization.py          #   风险平价 (ERC/MinVar/MaxDiv)
    ├── test_output.py                #   JSON 格式化和校验
    ├── test_rl.py                    #   强化学习 (环境/DQN/特征/训练)
    ├── test_screening.py             #   筛选过滤器
    ├── test_strategies.py            #   多策略 (动量/回归/质量/情绪/引擎)
    ├── test_trading_calendar.py      #   交易日历
    └── test_transformer.py           #   Transformer (31 tests)
```
