"""產生 RECORDED-DATA 的資料副本、表格與圖。

用法（在 RECORDED-DATA 資料夾裡執行）：
    python analysis/make_report.py
    python analysis/make_report.py --src ../Paper-trading --backtest B:/qfin-main/IEEE_EXT/D

- --src：雲端 repo（jaison5/Paper-trading）的本機副本，先 git pull。
- --backtest：論文回測的資料夾（只讀取 results/d6_weights_QQQ.csv 和 data/rt_*.csv，不寫入）。
  找不到時跳過回測對照（第 6、7 節）。
只需要 numpy、pandas；圖是手寫的 SVG，不需要 matplotlib。
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TD = 252
STRATS = ["E2E-QLSTM", "QVT-QLSTM", "QVT-QFWP", "VT-20d"]
EXEC_LABEL = {"model": "模型（收盤價成交）", "open": "券商：隔日開盤", "close": "券商：收盤前 5 分鐘", "bh": "買進持有"}
COLOR = {"E2E-QLSTM": "#0072B2", "QVT-QLSTM": "#D55E00", "QVT-QFWP": "#CC79A7", "VT-20d": "#009E73",
         "QQQ": "#333333", "VOO": "#8C8C8C", "IVV": "#8C8C8C", "SPY": "#8C8C8C"}
EXEC_COLOR = {"model": "#5C5C5C", "open": "#0072B2", "close": "#56B4E9"}
INK, INK2, GRID, AXIS = "#1F1F1F", "#5C5C5C", "#E6E6E6", "#BDBDBD"
FONT = "'Microsoft JhengHei','PingFang TC','Noto Sans TC','Noto Sans CJK TC',sans-serif"

sys.stdout.reconfigure(encoding="utf-8")


# ------------------------------------------------------------------ 0. 資料副本

def sync(src: Path):
    d = ROOT / "data"
    (d / "broker_orders").mkdir(parents=True, exist_ok=True)
    pairs = {"reports/daily_returns.csv": "daily_returns.csv", "reports/summary.md": "summary_auto.md",
             "ledger/signals.csv": "signals.csv", "ledger/orders.csv": "orders.csv",
             "ledger/snapshots.csv": "account_snapshots.csv", "ledger/cloud_runs.log": "cloud_runs.log",
             "ledger/freeze.json": "freeze.json"}
    for a, b in pairs.items():
        shutil.copy2(src / a, d / b)
    for f in sorted((src / "ledger").glob("broker_orders_*.csv")):
        shutil.copy2(f, d / "broker_orders" / f.name)
    shutil.copy2(src / "PROTOCOL.md", ROOT / "PROTOCOL.md")


def parse_col(c: str):
    """daily_returns.csv 的欄名 → (名稱, 執行方式)。"""
    if c.endswith("buy & hold"):
        return c.split()[0], "bh"
    if "(model, close fills)" in c:
        return c.split(" (")[0], "model"
    name = c.split(" (")[0]
    if name.endswith("-close"):
        return name[:-6], "close"
    return name, "open"


# ------------------------------------------------------------------ 1. 績效

def perf(r: pd.Series) -> dict:
    r = r.dropna()
    eq = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    return dict(days=len(r), total=eq.iloc[-1] - 1, vol=r.std() * math.sqrt(TD),
                mdd=(eq / eq.cummax() - 1).min(), worst=r.min())


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) + " |")
    return "\n".join(out)


def pct(v, d=1, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v * 100:+.{d}f}%" if sign else f"{v * 100:.{d}f}%"


# ------------------------------------------------------------------ SVG

class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []

    def add(self, s):
        self.el.append(s)

    def text(self, x, y, s, size=12, color=INK, anchor="start", weight="normal"):
        s = str(s).replace("&", "&amp;").replace("<", "&lt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                 f'font-weight="{weight}">{s}</text>')

    def line(self, x1, y1, x2, y2, color=AXIS, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')

    def save(self, path: Path, title: str):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" height="{self.h}" '
                f'font-family="{FONT}"><title>{title}</title><rect width="100%" height="100%" fill="#FFFFFF"/>')
        path.write_text(head + "\n".join(self.el) + "</svg>\n", encoding="utf-8")


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    v, out = math.ceil(lo / step) * step, []
    while v <= hi + 1e-12:
        out.append(round(v, 10))
        v += step
    return out


class Panel:
    def __init__(self, svg, x0, y0, w, h, xlim, ylim):
        self.s, self.x0, self.y0, self.w, self.h, self.xlim, self.ylim = svg, x0, y0, w, h, xlim, ylim

    def X(self, v):
        return self.x0 + (v - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

    def Y(self, v):
        return self.y0 + self.h - (v - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

    def ygrid(self, ticks, fmt):
        for t in ticks:
            y = self.Y(t)
            self.s.line(self.x0, y, self.x0 + self.w, y, GRID if abs(t) > 1e-12 else AXIS, 1)
            self.s.text(self.x0 - 6, y + 4, fmt(t), 11, INK2, "end")

    def xgrid(self, ticks, fmt):
        for t in ticks:
            x = self.X(t)
            self.s.line(x, self.y0, x, self.y0 + self.h, GRID if abs(t) > 1e-12 else AXIS, 1)
            self.s.text(x, self.y0 + self.h + 16, fmt(t), 11, INK2, "middle")

    def xlabels(self, pos, labels):
        for p, lab in zip(pos, labels):
            x = self.X(p)
            self.s.line(x, self.y0 + self.h, x, self.y0 + self.h + 4, AXIS)
            self.s.text(x, self.y0 + self.h + 17, lab, 11, INK2, "middle")

    def poly(self, xs, ys, color, width=2, dash=None, tip=None):
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys) if not np.isnan(y))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        t = f"<title>{tip}</title>" if tip else ""
        self.s.add(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                   f'stroke-linejoin="round" stroke-linecap="round"{d}>{t}</polyline>')


def spread(ys, gap):
    """直接標籤的位置往下推，避免重疊（ys 是像素座標）。"""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def end_labels(p: Panel, items, x_end, gap=15):
    """items: (標籤, 最後 y 值, 顏色)。"""
    ys = spread([p.Y(v) for _, v, _ in items], gap)
    for (lab, v, c), y in zip(items, ys):
        p.s.line(p.X(x_end) + 3, p.Y(v), p.X(x_end) + 12, y, c, 1)
        p.s.text(p.X(x_end) + 15, y + 4, lab, 12, INK)


def date_ticks(dates):
    """每週一個刻度。"""
    pos, lab, last = [], [], None
    for i, d in enumerate(dates):
        wk = d.isocalendar()[1]
        if wk != last:
            pos.append(i)
            lab.append(d.strftime("%m-%d"))
            last = wk
    return pos, lab


# ------------------------------------------------------------------ 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT.parent / "Paper-trading"))
    ap.add_argument("--backtest", default="B:/qfin-main/IEEE_EXT/D")
    ap.add_argument("--no-sync", action="store_true")
    a = ap.parse_args()
    src, btdir = Path(a.src), Path(a.backtest)
    if not a.no_sync:
        sync(src)
    D, T, F = ROOT / "data", ROOT / "tables", ROOT / "figures"
    T.mkdir(exist_ok=True)
    F.mkdir(exist_ok=True)
    md = [f"# 自動產生的表格（{pd.Timestamp.now():%Y-%m-%d %H:%M}，`analysis/make_report.py`）\n",
          "所有數字都由 `data/` 的紀錄算出；年化用 252 天。觀察期很短，只做描述，不做顯著性推論。\n"]

    R = pd.read_csv(D / "daily_returns.csv", index_col=0, parse_dates=True)
    meta = {c: parse_col(c) for c in R.columns}
    start, end = R.index[0], R.index[-1]
    q = perf(R["QQQ buy & hold"])

    # ---- 1. 績效
    rows = []
    for c in R.columns:
        n, e = meta[c]
        s = perf(R[c])
        rows.append(dict(name=n, exec=e, **s))
    P = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(STRATS + ["VOO", "IVV", "SPY", "QQQ"])}
    eorder = {"model": 0, "open": 1, "close": 2, "bh": 3}
    P = P.sort_values(by=["name", "exec"], key=lambda s: s.map(order) if s.name == "name" else s.map(eorder))
    P["vol_vs_qqq"] = P["vol"] / q["vol"]
    P["loss_vs_qqq"] = P["total"] / q["total"]
    P.to_csv(T / "performance.csv", index=False, float_format="%.6f")
    show = pd.DataFrame({"策略": P["name"], "執行方式": P["exec"].map(EXEC_LABEL), "天數": P["days"],
                         "累積報酬": P["total"].map(lambda v: pct(v, 2)), "年化波動": P["vol"].map(pct),
                         "波動÷QQQ": P["vol_vs_qqq"].map(lambda v: f"{v:.2f}"),
                         "虧損÷QQQ 虧損": P["loss_vs_qqq"].map(lambda v: f"{v:.2f}"),
                         "最大回撤": P["mdd"].map(pct), "最差一天": P["worst"].map(lambda v: pct(v, 2))})
    md += [f"## 1. 績效（{start:%Y-%m-%d}～{end:%Y-%m-%d}，{len(R)} 個交易日）\n", md_table(show), ""]

    # ---- 2. 執行差距
    bo = {}
    for f in sorted((D / "broker_orders").glob("broker_orders_*.csv")):
        acct = f.stem.replace("broker_orders_", "")
        bo[acct] = pd.read_csv(f)
    first_fill = {}
    for acct, df in bo.items():
        x = df[df["filled_qty"].fillna(0) > 0]
        first_fill[acct] = pd.Timestamp(x["filled_at"].min()[:10]) if len(x) else pd.NaT
    col = {(n, e): c for c, (n, e) in meta.items()}
    rows = []
    for s in STRATS:
        m = R[col[(s, "model")]]
        for e, acct in (("open", s), ("close", f"{s}-close")):
            if (s, e) not in col:
                continue
            b = R[col[(s, e)]]
            ff = first_fill.get(acct)
            after = R.index > ff  # 第一筆成交之後的完整交易日
            same_b = (1 + b[after]).prod() - 1
            same_m = (1 + m[after]).prod() - 1
            rows.append(dict(strategy=s, exec=e, first_fill=ff.date(), model_total=(1 + m).prod() - 1,
                             broker_total=(1 + b).prod() - 1, gap_full=(1 + b).prod() - (1 + m).prod(),
                             same_start=R.index[after][0].date(), model_same=same_m, broker_same=same_b,
                             gap_same=same_b - same_m, te_daily=(b[after] - m[after]).std() * math.sqrt(TD)))
    G = pd.DataFrame(rows)
    G.to_csv(T / "execution_gap.csv", index=False, float_format="%.6f")
    spread_strat = max(R[col[(s, "model")]].add(1).prod() for s in STRATS) - min(R[col[(s, "model")]].add(1).prod() for s in STRATS)
    show = pd.DataFrame({"策略": G["strategy"], "帳戶": G["exec"].map(EXEC_LABEL), "第一筆成交": G["first_fill"],
                         "全期：模型": G["model_total"].map(lambda v: pct(v, 2)), "全期：券商": G["broker_total"].map(lambda v: pct(v, 2)),
                         "全期差距": G["gap_full"].map(lambda v: pct(v, 2, True)),
                         "同起點": G["same_start"], "同起點：模型": G["model_same"].map(lambda v: pct(v, 2)),
                         "同起點：券商": G["broker_same"].map(lambda v: pct(v, 2)), "同起點差距": G["gap_same"].map(lambda v: pct(v, 2, True)),
                         "每日差距的年化標準差": G["te_daily"].map(pct)})
    md += ["## 2. 執行差距（券商帳戶 − 模型的收盤價複製）\n",
           "「全期」從 10-21 起算，收盤前帳戶 10-21 錯過、10-22 收盤才第一次成交，所以全期差距含了 10-22 那天（QQQ −2.86%）沒有持股。"
           "「同起點」只算第一筆成交之後的交易日，兩邊起點相同。\n",
           md_table(show), "",
           f"4 個策略之間（模型版）的累積報酬差距最大 {pct(spread_strat, 2)}。\n"]

    # ---- 3. 成交價偏離（只看 QQQ；BIL 每月配息，還原價和成交價不能直接比）
    px = pd.read_csv(src / "data" / "ohlc_QQQ.csv", parse_dates=["date"]).set_index("date")
    rows = []
    for acct, df in bo.items():
        x = df[(df["symbol"] == "QQQ") & (df["filled_qty"].fillna(0) > 0)].copy()
        if x.empty:
            continue
        x["day"] = pd.to_datetime(x["filled_at"].str[:10])
        e = "close" if acct.endswith("-close") else "open"
        ref = px["Close"] if e == "close" else px["Open"]
        x["ref"] = x["day"].map(ref)
        sign = np.where(x["side"] == "buy", 1, -1)
        x["bps"] = sign * (x["filled_avg_price"] - x["ref"]) / x["ref"] * 1e4  # 正值＝對自己不利
        for _, r in x.iterrows():
            rows.append(dict(account=acct, exec=e, day=r["day"].date(), side=r["side"], qty=r["filled_qty"],
                             fill=r["filled_avg_price"], ref=r["ref"], bps=r["bps"]))
    S = pd.DataFrame(rows)
    S.to_csv(T / "fill_vs_reference.csv", index=False, float_format="%.4f")
    agg = S.groupby("exec")["bps"].agg(["count", "mean", "median", lambda v: v.abs().max()]).reset_index()
    agg.columns = ["exec", "n", "mean", "median", "absmax"]
    show = pd.DataFrame({"帳戶": agg["exec"].map(EXEC_LABEL), "對照價": agg["exec"].map({"open": "Yahoo 當天開盤價", "close": "Yahoo 當天收盤價"}),
                         "QQQ 成交筆數": agg["n"], "平均偏離（bp）": agg["mean"].map(lambda v: f"{v:+.1f}"),
                         "中位數（bp）": agg["median"].map(lambda v: f"{v:+.1f}"), "最大絕對值（bp）": agg["absmax"].map(lambda v: f"{v:.1f}")})
    md += ["## 3. QQQ 成交價相對於官方價格的偏離\n", "正值代表對自己不利（買得比參考價貴、賣得比參考價便宜）。1 bp＝0.01%。\n",
           md_table(show), ""]

    # ---- 4. 訂單狀態與錯過的訊號
    rows = []
    for acct, df in bo.items():
        vc = df["status"].value_counts()
        rows.append(dict(帳戶=acct, 訂單數=len(df), 成交=int(vc.get("filled", 0)), 部分成交=int(vc.get("partially_filled", 0)),
                         過期=int(vc.get("expired", 0)), 取消=int(vc.get("canceled", 0)), 拒絕=int(vc.get("rejected", 0)),
                         排隊中=int(vc.get("accepted", 0) + vc.get("new", 0) + vc.get("pending_new", 0))))
    O = pd.DataFrame(rows)
    O.to_csv(T / "order_status.csv", index=False)
    sig = pd.read_csv(D / "signals.csv", parse_dates=["signal_date", "exec_session"])
    md += ["## 4. 訂單狀態（券商最後狀態）與訊號\n", md_table(O), "",
           f"- 訊號紀錄 `signals.csv`：{sig['signal_date'].nunique()} 個訊號日 × {sig['strategy'].nunique()} 個策略，"
           f"標記為錯過的 {int(sig['missed'].sum())} 筆。",
           "- 過期／取消的訂單都在 10-20～10-21 設定期間（收盤單不被撮合、改設定時取消），見 `PROTOCOL.md` 變更紀錄。",
           "- 收盤前帳戶 10-21 因 GitHub 排程延遲錯過，這件事只記在 `PROTOCOL.md` 和 `cloud_runs.log`，`signals.csv` 沒有逐帳戶的錯過欄位。", ""]

    # ---- 5. 事先登記的預期
    qv = q["vol"]
    vols = P[P["name"].isin(STRATS)]
    e2e = P[P["name"] == "E2E-QLSTM"]
    w = sig.pivot(index="exec_session", columns="strategy", values="target_w")
    vt3 = ["QVT-QLSTM", "QVT-QFWP", "VT-20d"]
    all_full = w[vt3].ge(0.9999).all(axis=1)
    full_days = [d for d in w.index[all_full] if d in R.index]
    same = all(abs(R.loc[d, col[("QVT-QLSTM", "model")]] - R.loc[d, col[(s, "model")]]) < 1e-12 for d in full_days for s in vt3)
    gap_open_max = G[G["exec"] == "open"]["gap_full"].abs().max()
    gap_same_max = G["gap_same"].abs().max()
    rows = [
        dict(預期="量子策略與 VT-20d 的實際波動 ≤ QQQ", 結果=f"{pct(vols['vol'].min())}～{pct(vols['vol'].max())}，QQQ {pct(qv)}",
             判定="符合" if (vols["vol"] <= qv).all() else "不符合"),
        dict(預期="E2E-QLSTM 的波動約為 QQQ 的 40%～50%", 結果="、".join(f"{v:.0%}" for v in e2e["vol_vs_qqq"]),
             判定="符合" if e2e["vol_vs_qqq"].between(0.40, 0.50).all() else "部分符合"),
        dict(預期="波動 < 20% 時三個 VT 策略都持有 100%、報酬相同；波動上升才分開",
             結果=f"三者都 100% 的執行日 {len(full_days)} 天（{', '.join(d.strftime('%m-%d') for d in full_days)}），這些天報酬{'完全相同' if same else '不同'}；之後 σ̂ 超過 20% 才分開",
             判定="符合" if same else "不符合"),
        dict(預期="券商與模型的差距遠小於策略之間的差異",
             結果=f"開盤帳戶全期差距最大 {pct(gap_open_max, 2)}；同起點差距最大 {pct(gap_same_max, 2)}；策略之間最大 {pct(spread_strat, 2)}",
             判定="符合" if max(gap_open_max, gap_same_max) < spread_strat / 3 else "不符合"),
    ]
    E = pd.DataFrame(rows)
    E.to_csv(T / "preregistered_expectations.csv", index=False)
    md += ["## 5. 事先登記的預期（`PROTOCOL.md`，2026-10-21 寫好）\n", md_table(E), ""]

    # ---- σ̂ 與部位路徑
    qpx = px["Close"]
    h20 = qpx.pct_change().rolling(20).std() * math.sqrt(TD)
    sig["hist20"] = sig["signal_date"].map(h20)
    chk = sig[sig["strategy"] == "VT-20d"]
    vt_err = (np.minimum(1, 0.2 / chk["hist20"]) - chk["target_w"]).abs().max()
    path = sig.pivot(index="signal_date", columns="strategy", values="sigma_hat")
    path["hist20"] = h20.reindex(path.index)
    out = pd.DataFrame({"signal_date": path.index.date, "exec_session": sig.groupby("signal_date")["exec_session"].first().dt.date.values})
    for s in ["QVT-QLSTM", "QVT-QFWP"]:
        out[f"sigma_{s}"] = path[s].values
    out["sigma_hist20"] = path["hist20"].values
    for s in STRATS:
        out[f"w_{s}"] = sig[sig["strategy"] == s].set_index("signal_date")["target_w"].reindex(path.index).values
    out.to_csv(T / "sigma_and_weights.csv", index=False, float_format="%.6f")
    md += ["## 附：每天的 σ̂ 與目標部位\n", f"`tables/sigma_and_weights.csv`。VT-20d 用過去 20 天波動重算，和紀錄的部位最大差 {vt_err:.1e}（確認公式正確）。\n"]

    fig_cumulative(R, col, F)
    fig_paths(out, F)
    fig_execution(G, P, F)

    # ---- 固定比例對照：E2E-QLSTM 的部位幾乎不動
    we = sig[sig["strategy"] == "E2E-QLSTM"]["target_w"]
    wbar = we.mean()
    bil = pd.read_csv(src / "data" / "ohlc_BIL.csv", parse_dates=["date"]).set_index("date")["Close"].pct_change()
    qret = R["QQQ buy & hold"]
    fixed_live = wbar * qret + (1 - wbar) * bil.reindex(R.index).fillna(0.0)
    fx = perf(fixed_live)
    e2 = perf(R[col[("E2E-QLSTM", "model")]])
    live_fixed = dict(wbar=wbar, wmin=we.min(), wmax=we.max(), fixed=fx, e2e=e2)

    # ---- 6、7. 回測對照
    if (btdir / "results" / "d6_weights_QQQ.csv").exists():
        md += backtest_section(btdir, R, col, T, F, live_fixed)
    else:
        md += ["## 6. 回測對照\n", f"找不到 {btdir}，略過。\n"]

    (T / "summary_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


# ------------------------------------------------------------------ 圖

def fig_cumulative(R, col, F):
    series = [(s, R[col[(s, "model")]]) for s in STRATS] + [("QQQ", R["QQQ buy & hold"]), ("VOO", R["VOO buy & hold"])]
    dates = [R.index[0] - pd.offsets.BDay(1)] + list(R.index)
    cum = {n: np.r_[0.0, (1 + r).cumprod().values - 1] for n, r in series}
    lo = min(v.min() for v in cum.values())
    hi = max(v.max() for v in cum.values())
    s = SVG(860, 440)
    s.text(24, 30, "實盤模擬一個月的累積報酬（模型版，收盤價成交、單邊成本 0.05%）", 16, INK, weight="bold")
    s.text(24, 50, f"{dates[0]:%Y-%m-%d} 收盤起算，到 {dates[-1]:%Y-%m-%d}；VOO 代表三大 ETF（IVV、SPY 幾乎重疊）", 12, INK2)
    p = Panel(s, 70, 75, 590, 310, (0, len(dates) - 1), (min(lo, -0.02) - 0.01, max(hi, 0.0) + 0.01))
    p.ygrid(nice_ticks(p.ylim[0], p.ylim[1], 6), lambda t: f"{t * 100:.0f}%".replace("-", "−"))
    pos, lab = date_ticks(dates)
    p.xlabels(pos, lab)
    xs = list(range(len(dates)))
    for n, _ in series:
        dash = "6 4" if n in ("QVT-QFWP", "VOO") else None
        width = 2.5 if n == "E2E-QLSTM" else 2
        p.poly(xs, cum[n], COLOR[n], width, dash, tip=f"{n} {cum[n][-1] * 100:+.1f}%")
    end_labels(p, [(f"{n} {cum[n][-1] * 100:.1f}%".replace("-", "−"), cum[n][-1], COLOR[n]) for n, _ in series], len(dates) - 1)
    s.text(24, 425, "資料：Alpaca 模擬帳戶的訊號紀錄與 Yahoo Finance 日線；年化與統計見 tables/summary_tables.md", 11, INK2)
    s.save(F / "fig1_cumulative.svg", "實盤模擬累積報酬")


def fig_paths(out, F):
    d = pd.to_datetime(out["signal_date"])
    n = len(d)
    s = SVG(860, 560)
    s.text(24, 30, "預測波動 σ̂ 與每天的目標 QQQ 部位（第 t 天收盤後算出）", 16, INK, weight="bold")
    s.text(24, 50, "上：σ̂ 超過 20%（虛線）才會減碼；下：部位 w ＝ min(1, 20% ÷ σ̂)，E2E-QLSTM 直接輸出部位", 12, INK2)
    xs = list(range(n))
    pos, lab = date_ticks(list(d))
    sig_cols = [("QVT-QLSTM", "sigma_QVT-QLSTM", "QLSTM 預測"), ("QVT-QFWP", "sigma_QVT-QFWP", "QFWP 預測"), ("VT-20d", "sigma_hist20", "過去 20 天")]
    hi = max(out[c].max() for _, c, _ in sig_cols)
    p = Panel(s, 70, 75, 590, 190, (0, n - 1), (0.10, max(0.45, hi + 0.02)))
    p.ygrid(nice_ticks(p.ylim[0], p.ylim[1], 5), lambda t: f"{t * 100:.0f}%")
    p.s.line(p.x0, p.Y(0.20), p.x0 + p.w, p.Y(0.20), INK2, 1, "4 3")
    p.xlabels(pos, lab)
    for k, c, name in sig_cols:
        p.poly(xs, out[c].values, COLOR[k], 2, "6 4" if k == "QVT-QFWP" else None, tip=name)
    end_labels(p, [(f"{name} {out[c].iloc[-1] * 100:.0f}%", out[c].iloc[-1], COLOR[k]) for k, c, name in sig_cols], n - 1)
    p2 = Panel(s, 70, 320, 590, 190, (0, n - 1), (0.0, 1.05))
    p2.ygrid([0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t * 100:.0f}%")
    p2.xlabels(pos, lab)
    for k in STRATS:
        p2.poly(xs, out[f"w_{k}"].values, COLOR[k], 2, "6 4" if k == "QVT-QFWP" else None, tip=k)
    end_labels(p2, [(f"{k} {out[f'w_{k}'].iloc[-1] * 100:.0f}%", out[f"w_{k}"].iloc[-1], COLOR[k]) for k in STRATS], n - 1)
    s.text(70, 308, "目標 QQQ 部位", 12, INK, weight="bold")
    s.text(70, 545, "橫軸是訊號日；部位在下一個交易日成交。資料：ledger/signals.csv", 11, INK2)
    s.save(F / "fig2_sigma_weight.svg", "預測波動與目標部位")


def fig_execution(G, P, F):
    model = P[P["exec"] == "model"].set_index("name")["total"]
    q = P[(P["name"] == "QQQ")]["total"].iloc[0]
    voo = P[(P["name"] == "VOO")]["total"].iloc[0]
    rows = []
    for st in STRATS:
        rows.append((st, "model", model[st]))
        for e in ("open", "close"):
            g = G[(G["strategy"] == st) & (G["exec"] == e)]
            if len(g):
                rows.append((st, e, g["broker_total"].iloc[0]))
    s = SVG(860, 470)
    s.text(24, 30, "同一個訊號、三種成交方式的累積報酬（10-21～11-20）", 16, INK, weight="bold")
    s.text(24, 50, "收盤前帳戶 10-21 錯過、10-22 收盤才進場，避開了 10-22 的 −2.86%，所以全期看起來比較好", 12, INK2)
    lo = min(min(v for *_, v in rows), q) - 0.02
    p = Panel(s, 190, 80, 560, 330, (lo, 0.0), (0, 1))
    p.xgrid(nice_ticks(lo, 0.0, 6), lambda t: f"{t * 100:.0f}%".replace("-", "−"))
    bar_h, gap_g = 18, 16
    y = p.y0 + 8
    for st in STRATS:
        s.text(p.x0 - 10, y + 1.5 * bar_h + 6, st, 13, INK, "end", "bold")
        for e in ("model", "open", "close"):
            v = [r for r in rows if r[0] == st and r[1] == e]
            if not v:
                continue
            v = v[0][2]
            x0, x1 = p.X(v), p.X(0)
            s.add(f'<rect x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="{bar_h - 2}" rx="3" fill="{EXEC_COLOR[e]}">'
                  f'<title>{st}／{EXEC_LABEL[e]}：{v * 100:.2f}%</title></rect>')
            s.text(x0 + 6, y + bar_h - 5, f"{v * 100:.1f}%".replace("-", "−"), 11, INK if e == "close" else "#FFFFFF")
            y += bar_h
        y += gap_g
    for v, name in ((q, "QQQ"), (voo, "VOO")):
        s.line(p.X(v), p.y0, p.X(v), p.y0 + p.h, INK if name == "QQQ" else INK2, 1.5, "5 3")
        s.text(p.X(v) + 4, p.y0 + 10, f"{name} {v * 100:.1f}%".replace("-", "−"), 11, INK2)
    lx = 190
    for e in ("model", "open", "close"):
        s.add(f'<rect x="{lx}" y="436" width="12" height="12" rx="2" fill="{EXEC_COLOR[e]}"/>')
        s.text(lx + 17, 446, EXEC_LABEL[e], 12, INK)
        lx += 175
    s.save(F / "fig3_execution.svg", "三種成交方式的累積報酬")
    return None


def backtest_section(btdir, R, col, T, F, live_fixed):
    W = pd.read_csv(btdir / "results" / "d6_weights_QQQ.csv", index_col=0, parse_dates=True)

    def close(t):
        df = pd.read_csv(btdir / "data" / f"rt_{t}.csv", parse_dates=["date"]).set_index("date").sort_index()
        return df[~df.index.duplicated()]["Close"]

    r = close("QQQ").pct_change()
    rf = close("IRX").reindex(r.index).ffill() / 100 / TD
    idx = W.index
    out = {}
    for name, c in [("E2E-QLSTM", "D6_QLSTM_cap1"), ("QVT-QLSTM", "D4_QLSTM_cap1"), ("QVT-QFWP", "D4_QFWP_cap1"), ("VT-20d", "plainVT_cap1")]:
        w = W[c]
        turn = w.diff().abs().fillna(w.abs())
        cash = 1 - w
        out[name] = w * r.reindex(idx) + cash * rf.reindex(idx) - np.maximum(-cash, 0) * 0.005 / TD - 0.0005 * turn
    out["QQQ"] = r.reindex(idx)
    out["SPY"] = close("SPY").pct_change().reindex(idx)
    B = pd.DataFrame(out)

    # 重現檢查：論文的 CAGR÷波動
    rep = []
    for n in B.columns:
        x = B[n].dropna()
        eq = (1 + x).cumprod()
        yrs = len(x) / TD
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        vol = x.std() * math.sqrt(TD)
        rep.append(dict(策略=n, 期間=f"{x.index[0]:%Y-%m-%d}～{x.index[-1]:%Y-%m-%d}", CAGR=pct(cagr), 年化波動=pct(vol),
                        **{"CAGR÷波動": f"{cagr / vol:.2f}"}, 最大回撤=pct((eq / eq.cummax() - 1).min())))
    # 固定比例對照：持有 E2E-QLSTM 平均部位的 QQQ，其餘國庫券，每天再平衡、不計成本
    ew = W["D6_QLSTM_cap1"]
    wb = ew.mean()
    fixed = (wb * r.reindex(idx) + (1 - wb) * rf.reindex(idx)).dropna()
    eq = (1 + fixed).cumprod()
    yrs = len(fixed) / TD
    cagr, vol = eq.iloc[-1] ** (1 / yrs) - 1, fixed.std() * math.sqrt(TD)
    rep.append(dict(策略=f"固定 {wb:.0%} QQQ＋國庫券（事後用 E2E 的平均部位）", 期間=f"{fixed.index[0]:%Y-%m-%d}～{fixed.index[-1]:%Y-%m-%d}",
                    CAGR=pct(cagr), 年化波動=pct(vol), **{"CAGR÷波動": f"{cagr / vol:.2f}"}, 最大回撤=pct((eq / eq.cummax() - 1).min())))
    rep = pd.DataFrame(rep)
    lf = live_fixed
    fixed_md = ["## 8. E2E-QLSTM 的部位幾乎固定\n",
                f"- 本月：目標部位 {lf['wmin']:.1%}～{lf['wmax']:.1%}（平均 {lf['wbar']:.1%}）。"
                f"同期固定持有 {lf['wbar']:.1%} QQQ、其餘 BIL：累積 {pct(lf['fixed']['total'], 2)}、波動 {pct(lf['fixed']['vol'])}、最大回撤 {pct(lf['fixed']['mdd'])}；"
                f"E2E-QLSTM（模型版）是 {pct(lf['e2e']['total'], 2)}、{pct(lf['e2e']['vol'])}、{pct(lf['e2e']['mdd'])}。",
                f"- 回測 2008～2026：E2E-QLSTM 的部位平均 {wb:.1%}、標準差 {ew.std():.1%}、1%～99% 分位 {ew.quantile(0.01):.1%}～{ew.quantile(0.99):.1%}；"
                f"和 VT-20d 部位的相關 {np.corrcoef(ew, W['plainVT_cap1'])[0, 1]:.2f}。固定比例的全期數字在第 7 節最後一列。",
                "- 意思：本月「只虧 QQQ 的一半」幾乎完全來自平均只持有約 47% 的 QQQ，不是 E2E 在下跌時減碼。", ""]
    # 每年頭兩個交易日還是前一年模型的部位（決策到持有差兩天），算年內高低時排除
    inner = ew[ew.groupby(ew.index.year).cumcount() >= 2]
    yr = inner.groupby(inner.index.year).agg(["mean", "min", "max"])
    qy = (1 + B["QQQ"]).groupby(B.index.year).prod() - 1
    ey = (1 + B["E2E-QLSTM"]).groupby(B.index.year).prod() - 1
    yt = pd.DataFrame({"年": yr.index, "平均部位": yr["mean"].map(pct), "最低": yr["min"].map(pct), "最高": yr["max"].map(pct),
                       "最高−最低": (yr["max"] - yr["min"]).map(lambda v: f"{v * 100:.1f} pt"),
                       "QQQ 當年報酬": qy.map(pct).values, "E2E 當年報酬": ey.map(pct).values})
    yt.to_csv(T / "e2e_weight_by_year.csv", index=False)
    fixed_md += ["回測中 E2E-QLSTM 每年的部位（每年用前一年以前的資料重新訓練，所以年和年之間會跳）：\n", md_table(yt), ""]

    # 23 天滾動視窗
    L = len(R)
    roll = np.expm1(np.log1p(B).rolling(L).sum()).dropna()
    live = {n: (1 + R[col[(n, "model")]]).prod() - 1 for n in STRATS}
    live["QQQ"] = (1 + R["QQQ buy & hold"]).prod() - 1
    live["SPY"] = (1 + R["SPY buy & hold"]).prod() - 1
    down = roll[roll["QQQ"] <= -0.10]
    ends = np.flatnonzero(roll["QQQ"].values <= -0.10)
    episodes = 1 + int((np.diff(ends) > L).sum()) if len(ends) else 0
    rows = []
    for n in STRATS:
        ratio = down[n] / down["QQQ"]
        lr = live[n] / live["QQQ"]
        rows.append(dict(策略=n, **{"回測：少虧於 QQQ 的比例": f"{(down[n] > down['QQQ']).mean():.0%}",
                                   "回測：少虧於 SPY 的比例": f"{(down[n] > down['SPY']).mean():.0%}",
                                   "回測：虧損÷QQQ 虧損（中位數）": f"{ratio.median():.2f}",
                                   "回測：10%～90% 區間": f"{ratio.quantile(0.1):.2f}～{ratio.quantile(0.9):.2f}",
                                   "本月實盤：虧損÷QQQ 虧損": f"{lr:.2f}",
                                   "本月在回測分布的百分位": f"{(ratio <= lr).mean():.0%}",
                                   "本月少虧於 SPY": "是" if live[n] > live["SPY"] else "否"}))
    Wt = pd.DataFrame(rows)
    Wt.to_csv(T / "backtest_23d_windows.csv", index=False)
    fig_scatter(roll, live, L, F)
    return ["## 6. 和論文回測比：歷史上 QQQ 在 " + f"{L} 個交易日內跌超過 10% 的時候\n",
            f"用論文的回測部位（2008-01-02～2026-10-19，`d6_weights_QQQ.csv`）重建每日報酬，取所有 {L} 天的滾動視窗，"
            f"挑出 QQQ 同期跌超過 10% 的 {len(down)} 個視窗（彼此重疊，約 {episodes} 段獨立的下跌）。"
            "百分位越低代表本月比回測的典型情況少虧得越多。\n",
            md_table(Wt), "",
            "## 7. 論文回測重現檢查（全期，同論文 Table 的 CAGR÷波動）\n", md_table(rep), ""] + fixed_md


def fig_scatter(roll, live, L, F):
    sub = roll.iloc[::5]
    s = SVG(860, 720)
    s.text(24, 30, f"本月實盤落在回測的哪裡？（{L} 個交易日的報酬，2008～2026，每 5 天取一個視窗）", 16, INK, weight="bold")
    s.text(24, 50, "橫軸 QQQ、縱軸策略；虛線以下＝比 QQQ 虧更多。黑圈是本月實盤（模型版）。", 12, INK2)
    lo = min(roll["QQQ"].min(), min(roll[n].min() for n in STRATS)) - 0.02
    hi = max(roll["QQQ"].max(), max(roll[n].max() for n in STRATS)) + 0.02
    ticks = nice_ticks(lo, hi, 5)
    for k, n in enumerate(STRATS):
        cx, cy = 80 + (k % 2) * 400, 85 + (k // 2) * 315
        p = Panel(s, cx, cy, 330, 250, (lo, hi), (lo, hi))
        p.ygrid(ticks, lambda t: f"{t * 100:.0f}%".replace("-", "−"))
        p.xgrid(ticks, lambda t: f"{t * 100:.0f}%".replace("-", "−"))
        s.line(p.X(lo), p.Y(lo), p.X(hi), p.Y(hi), INK2, 1, "4 3")
        dots = "".join(f'<circle cx="{p.X(a):.1f}" cy="{p.Y(b):.1f}" r="2.2"/>' for a, b in zip(sub["QQQ"], sub[n]))
        s.add(f'<g fill="{COLOR[n]}" fill-opacity="0.28">{dots}</g>')
        lx, ly = p.X(live["QQQ"]), p.Y(live[n])
        s.add(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6.5" fill="#FFFFFF" stroke="{INK}" stroke-width="2.5">'
              f'<title>本月：QQQ {live["QQQ"] * 100:.1f}%，{n} {live[n] * 100:.1f}%</title></circle>')
        s.text(lx + 10, ly + 4, f"本月 {live[n] * 100:.1f}%".replace("-", "−"), 11, INK, weight="bold")
        s.text(cx, cy - 8, n, 13, INK, weight="bold")
    s.text(24, 705, "回測：論文的 2008～2026 逐年 walk-forward 部位；本月：實盤模擬的模型版（同樣的收盤價成交與成本）", 11, INK2)
    s.save(F / "fig4_backtest_windows.svg", "本月實盤與回測的比較")


if __name__ == "__main__":
    main()
