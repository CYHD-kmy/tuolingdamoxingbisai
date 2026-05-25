# 智投未来 — A股日内投资智能体

基于 LangGraph 多智能体协作的 A 股日内投资决策系统。

## 架构

```
数据层 (AKShare/Tushare/BaoStock) → 海选筛选 (8因子打分) → 多Agent分析 (4分析师) → 辩论决策 → JSON输出
```

## 快速开始

```bash
# 安装依赖
pip install akshare baostock requests

# 可选: Tushare (需要注册获取 Token)
pip install tushare

# 设置环境变量
export LLM_API_KEY=sk-xxx
export TUSHARE_TOKEN=your_token  # 可选

# 运行
python -m src.main
```

## 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 数据层 | `src/data/` | 多源降级编排 (AKShare → Tushare → BaoStock) |
| 海选筛选 | `src/screening/` | ST/停牌过滤 + 8因子加权打分 |
| LLM 适配 | `src/llm/` | OpenAI-compatible 客户端 (DeepSeek/OpenAI) |
| 分析Agent | `src/agents/` | 技术面/基本面/资金面/消息面 四维分析 |

## 策略

- **LLM 分层**: quick 模型 (分析师) + deep 模型 (决策主管)
- **风控**: 单票≤20%、行业≤40%、日内5%熔断
- **输出**: 赛道标准 JSON 格式 + Markdown 日报
