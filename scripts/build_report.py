from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml
import re
from glob import glob
import akshare as ak
import matplotlib.pyplot as plt
from urllib.error import URLError
from http.client import RemoteDisconnected
import time

# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    fund_code: str
    fund_name: str
    csindex_code: str
    lookback_days: int
    report_title: str
    output_dir: Path
    assets_dir: Path


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"])
    assets_dir = Path(cfg["assets_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        fund_code=str(cfg["fund_code"]),
        fund_name=str(cfg.get("fund_name", "")),
        csindex_code=str(cfg["csindex_code"]),
        lookback_days=int(cfg.get("lookback_days", 180)),
        report_title=str(cfg.get("report_title", "Weekly Report")),
        output_dir=out_dir,
        assets_dir=assets_dir,
    )


def load_manual_signals(path: str | Path) -> Dict[str, str]:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------
# Data fetch
# ----------------------------
def fetch_fund_nav_history(fund_code: str) -> pd.DataFrame:
    """
    AKShare: fund_open_fund_info_em(fund="007301", indicator="单位净值走势")
    返回字段通常包括: 净值日期/单位净值/日增长率 等。:contentReference[oaicite:5]{index=5}
    """
    try:
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    except TypeError:
        # 兼容极少数旧版本参数名为 fund 的情况
        df = ak.fund_open_fund_info_em(fund=fund_code, indicator="单位净值走势")

    # 统一字段名
    # 常见列: "净值日期", "单位净值", "日增长率"
    df = df.rename(columns={
        "净值日期": "date",
        "单位净值": "nav",
        "日增长率": "daily_pct",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df.dropna(subset=["nav"])


def fetch_fund_nav_snapshot(fund_code: str) -> pd.Series | None:
    """
    AKShare: fund_open_fund_daily_em() 返回所有开放式基金的当日净值快照。
    注意：该接口在交易日 16:00-23:00 更新。:contentReference[oaicite:6]{index=6}
    """
    df = ak.fund_open_fund_daily_em()
    row = df.loc[df["基金代码"] == fund_code]
    if row.empty:
        return None
    return row.iloc[0]


def fetch_csindex_valuation(csindex_code: str, retries: int = 3, sleep_s: float = 2.0) -> pd.DataFrame:
    """
    获取中证指数估值（可能是在线 Excel，偶发 RemoteDisconnected）。
    做简单重试，降低偶发失败。
    """
    last_err = None
    for i in range(retries):
        try:
            df = ak.stock_zh_index_value_csindex(symbol=csindex_code)
            df = df.rename(columns={
                "日期": "date",
                "市盈率1": "pe1",
                "市盈率2": "pe2",
                "股息率1": "dp1",
                "股息率2": "dp2",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            for c in ["pe1", "pe2", "dp1", "dp2"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except (URLError, RemoteDisconnected) as e:
            last_err = e
            time.sleep(sleep_s * (i + 1))
        except Exception as e:
            # 其他异常也先重试一次
            last_err = e
            time.sleep(sleep_s * (i + 1))

    raise RuntimeError(f"Failed to fetch csindex valuation after {retries} retries: {last_err}")


# ----------------------------
# Metrics
# ----------------------------

def calc_returns(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[-1 - periods] - 1.0)


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return float("nan")
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def percentile_rank(x: pd.Series, value: float) -> float:
    x = x.dropna()
    if x.empty or np.isnan(value):
        return float("nan")
    return float((x <= value).mean())


def score_direction(v: str) -> int:
    v = (v or "").strip().lower()
    if v == "up":
        return 1
    if v == "down":
        return -1
    if v == "flat":
        return 0
    return 0


def cycle_score(
    pe_percentile: float,
    momentum_4w: float,
    manual: Dict[str, str],
) -> Tuple[int, Dict[str, int]]:
    """
    简单、可解释的评分（-5 ~ +5）：
    - 估值：低估 +1，高估 -1，中性 0
    - 动量：4周收益>0 +1，否则 -1
    - 宏观方向（可选）：SIA/存储/CapEx/库存 每个 up +1, down -1
    """
    parts = {}

    # 估值分位：<30% 低估；>70% 高估
    if np.isnan(pe_percentile):
        parts["valuation"] = 0
    elif pe_percentile < 0.30:
        parts["valuation"] = 1
    elif pe_percentile > 0.70:
        parts["valuation"] = -1
    else:
        parts["valuation"] = 0

    # 动量
    if np.isnan(momentum_4w):
        parts["momentum"] = 0
    else:
        parts["momentum"] = 1 if momentum_4w > 0 else -1

    # 手工信号
    parts["sia"] = score_direction(manual.get("sia_sales_yoy", "unknown"))
    parts["memory"] = score_direction(manual.get("memory_price", "unknown"))
    parts["capex"] = score_direction(manual.get("capex", "unknown"))
    parts["inventory"] = score_direction(manual.get("inventory", "unknown"))

    total = int(sum(parts.values()))
    return total, parts


def recommendation(score: int, pe_percentile: float) -> str:
    """
    行为输出：只给两类动作（继续定投/暂停新增），避免高摩擦微操。
    - 分数偏负且估值处高位：暂停新增
    - 否则：继续定投
    """
    if (score <= -2) and (not np.isnan(pe_percentile)) and (pe_percentile > 0.70):
        return "暂停新增（保持现金缓冲，等待估值/动量回落后再恢复）"
    return "继续定投（保持周频与固定金额，不做短期赎回）"


# ----------------------------
# Plot + Markdown
# ----------------------------
def save_line_plot(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure()
    plt.plot(df[x], df[y])
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def build_markdown(
    cfg: Config,
    asof: date,
    nav_df: pd.DataFrame,
    snap: pd.Series | None,
    val_df: pd.DataFrame,
    score: int,
    score_parts: Dict[str, int],
    pe_percentile: float,
    mom_4w: float,
    mom_12w: float,
    dd: float,
    reco: str,
    assets: Dict[str, str],
    manual: Dict[str, str],
    val_fetch_error: str | None = None,
) -> str:
    last_nav_date = nav_df["date"].iloc[-1].date()
    last_nav = nav_df["nav"].iloc[-1]
    pe1 = float(val_df["pe1"].iloc[-1]) if ("pe1" in val_df.columns and not val_df["pe1"].empty) else float("nan")
    val_date = val_df["date"].iloc[-1].date() if not val_df.empty else None

    # 快照信息（若当日净值未更新则可能为空）
    snap_line = ""
    if snap is not None:
        snap_line = f"- 快照：单位净值={snap.get('单位净值','')}, 日增长率={snap.get('日增长率','')}（接口在交易日16:00-23:00更新）:contentReference[oaicite:8]{{index=8}}"

    notes = manual.get("notes", "")

    md = []
    md.append(f"# {cfg.report_title}")
    md.append(f"- 报告日期：{asof.isoformat()}")
    md.append(f"- 标的：{cfg.fund_name}（{cfg.fund_code}）:contentReference[oaicite:9]{{index=9}}")
    md.append(f"- 指数估值：中证全指半导体产品与设备指数（{cfg.csindex_code}）:contentReference[oaicite:10]{{index=10}}")
    md.append("")

    md.append("## 1) 本周核心结论（只给一个动作）")
    md.append(f"- **建议动作：{reco}**")
    md.append("")

    md.append("## 2) 关键观测量（状态变量）")
    if val_fetch_error:
        md.append(
            f"- 估值数据：本次获取失败（已降级输出，不影响净值、动量与回撤计算）。"
            f"错误摘要：{val_fetch_error[:120]}..."
        )
    md.append(f"- 最新净值日期：{last_nav_date}，单位净值：{last_nav:.4f}")
    if snap_line:
        md.append(snap_line)
    md.append(f"- 4周动量：{mom_4w*100:.2f}%；12周动量：{mom_12w*100:.2f}%")
    md.append(f"- 近{cfg.lookback_days}天最大回撤：{dd*100:.2f}%")
    if val_date is not None:
        md.append(f"- 估值（PE1）：{pe1:.2f}（日期：{val_date}），近{cfg.lookback_days}天分位：{pe_percentile*100:.1f}% :contentReference[oaicite:11]{{index=11}}")
    md.append("")

    md.append("## 3) 周期温度计（可解释评分）")
    md.append(f"- 总分：**{score}**（范围约 -5 ~ +5）")
    md.append(f"- 分项：{score_parts}")
    md.append("")
    md.append("评分口径：估值分位低=+1/高=-1；4周动量正=+1/负=-1；手工信号 up=+1/down=-1。")
    md.append("")

    md.append("## 4) 图表")
    if "nav_png" in assets:
        md.append(f"![NAV]({assets['nav_png']})")
    if "pe_png" in assets:
        md.append(f"![PE]({assets['pe_png']})")
    md.append("")

    md.append("## 5) 手工信号（可选外场）")
    md.append(f"- SIA 销售 YoY：{manual.get('sia_sales_yoy','unknown')}")
    md.append(f"- 存储价格：{manual.get('memory_price','unknown')}")
    md.append(f"- CapEx/设备：{manual.get('capex','unknown')}")
    md.append(f"- 库存：{manual.get('inventory','unknown')}")
    if notes:
        md.append(f"- 备注：{notes}")
    md.append("")

    md.append("## 6) 免责声明")
    md.append("本报告为学习与研究用途，不构成投资建议。你应始终以费用、赎回规则与自身风险承受能力为约束条件。")
    return "\n".join(md)
def _extract_kv_from_report(md_text: str) -> dict:
    """
    从单份周报 Markdown 中提取少量关键信息。
    通过正则做轻量解析，避免依赖 Markdown 解析库。
    """
    def m(pattern: str) -> str | None:
        mm = re.search(pattern, md_text, flags=re.MULTILINE)
        return mm.group(1).strip() if mm else None

    #  build_markdown() 里写过的字段
    report_date = m(r"^- 报告日期：(\d{4}-\d{2}-\d{2})\s*$")
    action = m(r"^- \*\*建议动作：(.+?)\*\*\s*$")
    score = m(r"^- 总分：\*\*(\-?\d+)\*\*")
    pe_pct = m(r"分位：([0-9.]+)%")
    mom_4w = m(r"^- 4周动量：([\-0-9.]+)%")
    dd = m(r"最大回撤：([\-0-9.]+)%")

    return {
        "report_date": report_date,
        "action": action,
        "score": score,
        "pe_pct": pe_pct,
        "mom_4w": mom_4w,
        "dd": dd,
    }


def update_index_md(reports_dir: Path, n_latest: int = 15) -> None:
    """
    汇总最近 n_latest 份 reports/*.md（排除 index.md），生成 reports/index.md
    """
    reports_dir = Path(reports_dir)
    index_path = reports_dir / "index.md"

    md_files = sorted(
        [p for p in reports_dir.glob("*.md") if p.name.lower() != "index.md"],
        key=lambda p: p.name,  # 你的文件名是 YYYY-MM-DD.md，按名字排序=按日期排序
        reverse=True,
    )

    md_files = md_files[:n_latest]

    rows = []
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = p.read_text(encoding="utf-8", errors="replace")

        kv = _extract_kv_from_report(text)
        link = f"[{p.stem}]({p.name})"
        rows.append({
            "date": kv.get("report_date") or p.stem,
            "link": link,
            "score": kv.get("score") or "",
            "action": kv.get("action") or "",
            "pe_pct": (kv.get("pe_pct") + "%") if kv.get("pe_pct") else "",
            "mom_4w": (kv.get("mom_4w") + "%") if kv.get("mom_4w") else "",
            "mdd": (kv.get("dd") + "%") if kv.get("dd") else "",
        })

    # 生成 index.md
    lines = []
    lines.append("# 半导体周期仪表盘 - 周报索引")
    lines.append("")
    lines.append(f"- 最近更新：{date.today().isoformat()}")
    lines.append(f"- 汇总范围：最近 {len(rows)} 份报告")
    lines.append("")
    lines.append("## 最近周报（倒序）")
    lines.append("")
    lines.append("| 日期 | 报告 | Score | 动作 | PE分位 | 4周动量 | 最大回撤 |")
    lines.append("|---|---|---:|---|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['date']} | {r['link']} | {r['score']} | {r['action']} | {r['pe_pct']} | {r['mom_4w']} | {r['mdd']} |"
        )

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {index_path}")


def main() -> None:
    cfg = load_config("config.yml")
    manual = load_manual_signals("data/manual_signals.yml")

    asof = date.today()

    nav = fetch_fund_nav_history(cfg.fund_code)
    # 截取 lookback
    nav = nav.loc[nav["date"] >= (pd.Timestamp(asof) - pd.Timedelta(days=cfg.lookback_days))].copy()

    snap = fetch_fund_nav_snapshot(cfg.fund_code)

    val_fetch_error = None
    try:
        val = fetch_csindex_valuation(cfg.csindex_code)
        val = val.loc[val["date"] >= (pd.Timestamp(asof) - pd.Timedelta(days=cfg.lookback_days))].copy()
    except Exception as e:
        val_fetch_error = str(e)
        val = pd.DataFrame(columns=["date", "pe1", "pe2", "dp1", "dp2"])

    nav_series = nav["nav"].reset_index(drop=True)

    # 用净值序列的“交易日步长”近似：4周≈20个交易日，12周≈60个交易日
    mom_4w = calc_returns(nav_series, periods=20)
    mom_12w = calc_returns(nav_series, periods=60)
    dd = max_drawdown(nav_series)

    pe_percentile = float("nan")
    if not val.empty and "pe1" in val.columns:
        pe_percentile = percentile_rank(val["pe1"], float(val["pe1"].iloc[-1]))

    score, parts = cycle_score(pe_percentile=pe_percentile, momentum_4w=mom_4w, manual=manual)
    reco = recommendation(score=score, pe_percentile=pe_percentile)

    # 图表
    assets = {}
    nav_png = cfg.assets_dir / f"nav_{asof.isoformat()}.png"
    save_line_plot(nav, "date", "nav", f"{cfg.fund_code} NAV (last {cfg.lookback_days}d)", nav_png)
    assets["nav_png"] = os.path.relpath(nav_png, cfg.output_dir)

    if not val.empty and "pe1" in val.columns:
        pe_png = cfg.assets_dir / f"pe_{asof.isoformat()}.png"
        save_line_plot(val.dropna(subset=["pe1"]), "date", "pe1", f"{cfg.csindex_code} PE1 (last {cfg.lookback_days}d)", pe_png)
        assets["pe_png"] = os.path.relpath(pe_png, cfg.output_dir)

    md = build_markdown(
        cfg=cfg,
        asof=asof,
        nav_df=nav,
        snap=snap,
        val_df=val,
        score=score,
        score_parts=parts,
        pe_percentile=pe_percentile,
        mom_4w=mom_4w,
        mom_12w=mom_12w,
        dd=dd,
        reco=reco,
        assets=assets,
        manual=manual,
        val_fetch_error=val_fetch_error,
        
    )

    out_path = cfg.output_dir / f"{asof.isoformat()}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"[ok] wrote {out_path}")


    # 更新 index.md
    update_index_md(cfg.output_dir, n_latest=15)


if __name__ == "__main__":
    main()


