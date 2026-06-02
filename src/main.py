"""
智投未来 — A股日内投资智能体 主入口。

使用方式:
    python -m src.main              # 正常模式 (需要数据源和 LLM API Key)
    python -m src.main --demo       # 演示模式 (使用样本数据，无需网络和 API)

环境变量:
    LLM_API_KEY      - LLM API Key (必填，--demo 模式下不需要)
    TUSHARE_TOKEN    - Tushare Token (可选，有则自动启用)
    LLM_QUICK_MODEL  - quick 模型 (默认 deepseek-chat)
    LLM_DEEP_MODEL   - deep 模型 (默认 deepseek-reasoner)
    LLM_BASE_URL     - API 地址 (默认 https://api.deepseek.com)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.config import get_config
from src.graph.workflow import run_pipeline
from src.output.json_formatter import format_decisions, validate_decisions
from src.output.trace_logger import save_trace
from src.output.report_generator import generate_daily_report


def setup_logging(level: int = logging.INFO) -> None:
    """配置日志"""
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    # 降低第三方库日志级别
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def main(demo: bool = False) -> None:
    """主函数"""
    setup_logging()
    logger = logging.getLogger("main")

    config = get_config()

    t0 = time.monotonic()

    if demo:
        from src.demo import generate_demo_state

        logger.info("===== 智投未来 启动 (演示模式) =====")
        logger.info("日期: %s", datetime.now().strftime("%Y-%m-%d"))
        logger.info("总资金: ¥%.0f", config.initial_capital)
        logger.info("模式: 演示数据 (无网络/LLM 调用)")
        logger.info("")

        state = generate_demo_state()
    else:
        # 前置检查
        if not config.llm_api_key:
            logger.error("LLM_API_KEY 未设置。请设置环境变量: export LLM_API_KEY=sk-xxx")
            logger.error("或使用演示模式: python -m src.main --demo")
            sys.exit(1)

        logger.info("===== 智投未来 启动 =====")
        logger.info("日期: %s", datetime.now().strftime("%Y-%m-%d"))
        logger.info("Quick LLM: %s", config.llm_quick)
        logger.info("Deep LLM:  %s", config.llm_deep)
        logger.info("Tushare: %s", "可用" if config.tushare_available else "未配置")
        logger.info("")

        # ── 跨日持仓加载 (在管道运行前) ──
        from src.agents.portfolio_tracker import PortfolioTracker

        tracker = PortfolioTracker(
            total_capital=config.initial_capital,
            results_dir=config.results_dir,
        )
        tracker.load()
        hold_cash = tracker.cash
        hold_positions = tracker.current_positions_dict()

        if tracker.tampered:
            logger.critical(
                "持仓文件校验失败! positions.json 被篡改或与管道决策记录不匹配。"
            )
            logger.critical(
                "系统将拒绝继续运行以保护账户安全。\n"
                "可能原因: 1) 文件被手动修改 2) 文件被其他来源的数据替换。\n"
                "恢复方法: 从 %s/backups/ 中找到正确的备份文件, 覆盖 positions.json",
                config.results_dir,
            )
            sys.exit(1)

        # 有效资金: 有持仓用实际权益, 无持仓用初始资金
        effective_capital = tracker.total_equity() if hold_positions else config.initial_capital
        if hold_positions:
            logger.info(
                "已加载持仓: %d 只, 总权益 ¥%.0f, 可用现金 ¥%.0f, 持仓市值 ¥%.0f",
                len(hold_positions), effective_capital, hold_cash,
                tracker.total_market_value(),
            )
        else:
            logger.info("无历史持仓，初始资金 ¥%.0f", hold_cash)
        logger.info("")

        try:
            state = run_pipeline(
                total_capital=effective_capital,
                available_cash=hold_cash,
                current_holdings=hold_positions,
            )
        except KeyboardInterrupt:
            logger.warning("用户中断")
            sys.exit(0)
        except Exception:
            logger.exception("流水线异常终止")
            sys.exit(1)

    total_elapsed = time.monotonic() - t0

    # ── 输出 ──────────────────────────────────

    print("\n" + "=" * 50)
    print("  最终决策 (赛道 JSON 格式)")
    print("=" * 50)

    final_result = getattr(state, "final_result", None)
    position_limits = getattr(state, "position_limits", {})
    daily_data = getattr(state, "daily_data", {})
    verdicts = getattr(state, "verdicts", {})
    errors = getattr(state, "errors", [])
    elapsed = getattr(state, "elapsed", {})

    # 使用实际可用现金 (跨日后不再是 50 万)
    actual_cash = tracker.cash if not demo else config.initial_capital
    if final_result and final_result.decisions:
        validated = validate_decisions(
            final_result.decisions,
            position_limits,
            daily_data,
            cash_available=actual_cash,
            min_cash_reserve=config.min_cash_reserve,
            total_capital=config.initial_capital,
            verdicts=verdicts,
        )
        final_result.decisions = validated
        decisions_json = format_decisions(validated)
        print(json.dumps(decisions_json, ensure_ascii=False, indent=2))
        print(f"\n使用资金: ¥{final_result.cash_used:,.0f}")
        print(f"剩余资金: ¥{final_result.cash_remaining:,.0f}")
    else:
        print("[]  (空仓)")

    print(f"\n总耗时: {total_elapsed:.1f}s")
    for stage, e in elapsed.items():
        print(f"  {stage}: {e:.1f}s")

    if errors:
        print(f"\n警告/错误 ({len(errors)}条):")
        for e in errors[:5]:
            print(f"  - {e}")

    # 保存结果
    if config.save_trace:
        trace_path = save_trace(state, total_elapsed, config.results_dir)
        logger.info("结果已保存: %s", trace_path)

        # ChromaDB 记忆索引 (异步最佳努力)
        try:
            from src.memory import MemoryStore
            memory = MemoryStore()
            if memory.available:
                with open(trace_path, encoding="utf-8") as f:
                    trace_data = json.load(f)
                memory.index_trace(trace_data)
                logger.info("记忆已索引: %d 条记录", memory.count())
        except Exception:
            logger.debug("记忆索引跳过 (chromadb 不可用或索引失败)", exc_info=True)

        # 持久化持仓 (仅非 demo 模式; demo 模式每次独立运行不跨日累加)
        if not demo:
            # 复用已加载的 tracker (含历史持仓 + 现金), 应用当日决策
            _final = getattr(state, "final_result", None)
            _daily = getattr(state, "daily_data", {})
            all_decisions = _final.decisions if _final else []

            # 分离买卖决策: 先卖后买 (释放现金给买入用)
            sell_decisions = [d for d in all_decisions if getattr(d, "direction", "buy") == "sell"]
            buy_decisions = [d for d in all_decisions if getattr(d, "direction", "buy") == "buy"]

            if sell_decisions:
                sold_amount = tracker.apply_sells(sell_decisions, _daily)
                logger.info("已卖出 %d 笔, 回收资金 ¥%.0f, 现金余额 ¥%.0f",
                            len(sell_decisions), sold_amount, tracker.cash)

            if buy_decisions:
                tracker.apply_decisions(buy_decisions, _daily)

            tracker.update_prices(_daily)
            tracker.record_daily()
            tracker.save()
            total_equity = tracker.total_equity()
            total_return = tracker.total_return()
            logger.info("持仓已保存: %.0f%% 仓位, 权益 ¥%.0f, 累计收益 %+.2f%%",
                        tracker.total_market_value() / total_equity * 100 if total_equity > 0 else 0,
                        total_equity, total_return)

            # 回写实际权益到 trace JSON, 使 API 看板展示正确数值
            try:
                with open(trace_path, encoding="utf-8") as f:
                    trace_data = json.load(f)
                trace_data["total_equity"] = round(total_equity, 2)
                trace_data["total_return"] = round(total_return, 2)
                with open(trace_path, "w", encoding="utf-8") as f:
                    json.dump(trace_data, f, ensure_ascii=False, indent=2)
            except Exception:
                logger.debug("回写权益到 trace 失败", exc_info=True)

            report_md = generate_daily_report(state, tracker)
            date_str = datetime.now().strftime("%Y%m%d")
            report_path = os.path.join(config.results_dir, f"report_{date_str}.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_md)
            logger.info("日报已保存: %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智投未来 — A股日内投资智能体")
    parser.add_argument("--demo", action="store_true", help="使用演示数据运行 (无需网络和 LLM API Key)")
    parser.add_argument("--strategy", default="default",
                        help="策略选择: default / momentum / mean_reversion / quality / sentiment / all")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"),
                        help="回测模式: 指定起始和结束日期 YYYYMMDD")
    parser.add_argument("--benchmark", default="000300", help="回测基准指数代码 (默认 000300)")
    parser.add_argument("--rl-train", action="store_true", help="训练 RL 模型")
    parser.add_argument("--rl-model", type=str, default="", help="RL 模型路径 (推断模式)")
    parser.add_argument("--rl-episodes", type=int, default=200, help="RL 训练轮数")
    parser.add_argument("--transformer-train", action="store_true", help="训练 Transformer 模型")
    parser.add_argument("--transformer-model", type=str, default="",
                        help="Transformer 模型路径 (推断模式)")
    parser.add_argument("--transformer-epochs", type=int, default=50,
                        help="Transformer 训练轮数")
    args = parser.parse_args()

    if args.backtest:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("backtest")
        from src.backtesting import BacktestEngine, BacktestConfig
        from src.backtesting.report import generate_backtest_report

        config = BacktestConfig(
            start_date=args.backtest[0],
            end_date=args.backtest[1],
            initial_capital=get_config().initial_capital,
            benchmark=args.benchmark,
        )
        engine = BacktestEngine(config)
        result = engine.run()
        report_prefix = generate_backtest_report(result, get_config().backtest_output_dir)
        logger.info("回测完成，报告已保存: %s", report_prefix)
        print(f"\n===== 回测绩效 =====")
        print(f"总收益率:   {result.metrics.total_return * 100:+.2f}%")
        print(f"年化收益率: {result.metrics.annualized_return * 100:+.2f}%")
        print(f"Sharpe:     {result.metrics.sharpe_ratio:.2f}")
        print(f"最大回撤:   {result.metrics.max_drawdown * 100:.2f}%")
        print(f"Calmar:     {result.metrics.calmar_ratio:.2f}")
        print(f"胜率:       {result.metrics.win_rate * 100:.1f}%")
        print(f"盈亏比:     {result.metrics.profit_factor:.2f}")
        print(f"最终权益:   ¥{result.final_equity:,.0f}")
        sys.exit(0)

    if args.rl_train:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("rl")
        from src.rl.agent import DQNAgent
        from src.rl.trainer import train_rl_agent
        from src.demo import _make_daily_data, _SAMPLE_STOCKS

        logger.info("开始 RL 训练 (%d episodes)...", args.rl_episodes)
        daily_data = _make_daily_data()
        codes = [s[0] for s in _SAMPLE_STOCKS]
        agent = train_rl_agent(codes, daily_data, episodes=args.rl_episodes)
        model_path = args.rl_model or os.path.join(get_config().results_dir, "rl_model.json")
        agent.save(model_path)
        logger.info("RL 模型已保存: %s", model_path)
        logger.info("训练完成: epsilon=%.4f total_reward=%.2f",
                    agent.epsilon, agent._total_reward)
        sys.exit(0)

    if args.transformer_train:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        logger = logging.getLogger("transformer")
        from src.transformer import StockTransformer, generate_training_data, train_transformer
        from src.demo import _SAMPLE_STOCKS

        logger.info("开始 Transformer 训练 (%d epochs)...", args.transformer_epochs)

        # 生成 60 天样本数据
        records_list: dict[str, list] = {}
        for code, name, price, pct, vol, turn, amt, pe, mv in _SAMPLE_STOCKS:
            records = []
            cur_price = price * 0.82
            for i in range(60):
                daily_pct = 0.6 + (i % 5) * 0.3
                cur_price = cur_price * (1 + daily_pct / 100)
                record = type("Record", (), {
                    "date": f"2026-05-{i+1:02d}",
                    "open": cur_price * 0.99, "high": cur_price * 1.02,
                    "low": cur_price * 0.98, "close": cur_price, "volume": 2e7,
                    "amount": cur_price * 2e7 * 0.7, "pct_chg": daily_pct,
                    "turnover": 3.0, "ma5": cur_price * 0.99,
                    "ma10": cur_price * 0.97, "ma20": cur_price * 0.95,
                    "macd_dif": 0.5, "macd_dea": 0.3, "macd_bar": 0.2,
                    "rsi_6": 55.0, "rsi_14": 52.0,
                })()
                records.append(record)
            records_list[code] = records

        samples = generate_training_data(
            records_list,
            seq_len=get_config().transformer_seq_len,
            forward_days=get_config().transformer_forward_days,
            max_seq_len=60,
        )
        logger.info("训练样本: %d 条", len(samples))

        if len(samples) == 0:
            logger.error("训练样本为空，请检查数据")
            sys.exit(1)

        model = StockTransformer()
        losses = train_transformer(
            model, samples,
            epochs=args.transformer_epochs,
            lr=get_config().transformer_lr,
        )
        model_path = args.transformer_model or os.path.join(
            get_config().results_dir, "transformer_model.json"
        )
        model.save(model_path)
        logger.info("Transformer 模型已保存: %s (final loss=%.6f)", model_path, losses[-1])
        sys.exit(0)

    main(demo=args.demo)
