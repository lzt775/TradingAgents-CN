"""Bind original TradingAgents tools and ETF archive lookups to one dated research run."""
from datetime import date, datetime
from pathlib import Path
import os
import re
import runpy

from core.tools.context import research_snapshot
from core.tools.registry import ToolRegistry
from langchain_core.utils.function_calling import convert_to_openai_tool
from quantumn.data.etf_market_context import breadth_context, index_context, sector_context
from quantumn.data.etf_research_feeds import FEEDS, feed_result
from quantumn.data.etf_research_facts import lookup, price_breaks, price_series

MODULES = {
    "fundamentals/etf_fundamentals": ("get_etf_fundamentals",),
    "fundamentals/financial_tools": ("get_income_analysis", "get_cashflow_analysis"),
    "market/technical_indicators": ("get_technical_indicators",),
    "news/stock_news": ("get_stock_news_headlines_unified", "get_stock_news_details_unified"),
    "market/index_market_tools": ("get_index_data", "get_market_breadth"),
    "market/sector_market_tools": ("get_sector_daily_tool",),
}
FILINGS = MODULES["fundamentals/financial_tools"]
NEWS = MODULES["news/stock_news"]
WINDOWS = {"close_5_sma": 5, "close_10_sma": 10, "close_20_sma": 20, "rsi_14": 15,
           "macd": 35, "macds": 35, "macdh": 35, "boll": 20, "boll_ub": 20, "boll_lb": 20}
MARKET_CONTEXT_TOOLS = {"get_index_data", "get_market_breadth", "get_sector_daily_tool"}
ROLE_TOOLS = {
    "etf_analyst_v2": ("lookup_market", "get_etf_fundamentals", "get_research_feed"),
    "market_analyst_v2": ("lookup_market", "get_technical_indicators", "get_market_breadth"),
    "news_analyst_v2": ("lookup_news", *NEWS, "get_research_feed"),
    "index": ("lookup_market", "get_index_data", "get_market_breadth"),
    "sector": ("get_sector_daily_tool", *NEWS),
    "financials": ("lookup_filing", *FILINGS, *NEWS, "get_research_feed"),
    "sentiment": (*NEWS, "get_market_breadth"),
}
DESCRIPTIONS = {
    "lookup_market": "核对程序事实（空字符串），或按日期查询行情、季报、规模、份额、拆分。",
    "lookup_news": "按政策、事件关键词或公司名称检索归档及候选新闻。",
    "lookup_filing": "按已披露持仓六位股票代码查财报原文摘录、页码和链接。",
}


class ResearchTools:
    def __init__(self, bundle):
        self.bundle = bundle
        self.cutoff = datetime.fromisoformat(bundle["as_of"])
        self.day = self.cutoff.date().isoformat()
        self.holdings = {h["stock"]: h.get("name", "") for h in
                         (bundle.get("disclosed_holdings") or {}).get("holdings", [])}
        self.native = {}
        self.schemas = {}
        self.news_cache = {}
        # The standalone worker loads only its selected registered tools. Loading the
        # platform-wide catalogue also initializes unrelated app and database providers.
        ToolRegistry(discover=False)
        root = Path(__file__).resolve().parents[1] / "core/tools/implementations"
        for module, names in MODULES.items():
            exports = runpy.run_path(str(root / (module + ".py")))
            if module == "news/stock_news":
                self.fetch_news = exports["_fetch_akshare_news_live"]
            for name in names:
                obj = exports[name]
                self.native[name] = obj
                spec = convert_to_openai_tool(obj)
                spec["function"]["description"] = spec["function"]["description"].split("\n")[0]
                if name == "get_technical_indicators":
                    spec["function"]["parameters"]["properties"]["indicators"]["description"] = (
                        "当前证据为收盘价，可用指标：" + ",".join(WINDOWS))
                if name == "get_market_breadth":
                    spec["function"]["description"] += "。可返回沪深300成分股20日均线广度，注意样本和最新日期。"
                if name == "get_sector_daily_tool":
                    spec["function"]["parameters"]["properties"]["sector_name"]["description"] = "同花顺行业全名，如半导体、通信设备。"
                self.schemas[name] = spec
        for name, description in DESCRIPTIONS.items():
            self.schemas[name] = {"type": "function", "function": {"name": name, "description": description,
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
        self.schemas["get_research_feed"] = {"type": "function", "function": {
            "name": "get_research_feed", "description": "按渠道查 ETF 行情、净值、新闻、财务指标或海外公司资讯。",
            "parameters": {"type": "object", "required": ["feed"], "properties": {
                "feed": {"type": "string", "enum": list(FEEDS), "description": "；".join(FEEDS.values())},
                "symbol": {"type": "string", "description": "海外新闻必须填写产业链公司代码，如 NVDA。"}}}}}

    def names(self, role):
        if role == "joint_research":
            return list(dict.fromkeys(name for names in ROLE_TOOLS.values() for name in names))
        return list(ROLE_TOOLS.get(role, ()))

    def preflight(self, role):
        return self.default_query(role, self.bundle["ticker"], self.day)

    @staticmethod
    def default_query(role, ticker, day):
        if role == "etf_analyst_v2":
            return "get_etf_fundamentals", dict(ticker=ticker, curr_date=day)
        if role == "market_analyst_v2":
            return "get_technical_indicators", dict(ticker=ticker, indicators="close_5_sma,close_20_sma,rsi_14")
        if role == "news_analyst_v2":
            return "get_stock_news_headlines_unified", dict(ticker=ticker, curr_date=day)
        if role == "index":
            return "get_index_data", dict(trade_date=day)
        if role == "sentiment":
            return "get_market_breadth", dict(trade_date=day)
        # These are broad industry comparators; they do not replace each fund's tracking index.
        sector = {"588170": "半导体", "515050": "通信设备"}.get(ticker)
        if role == "sector" and sector:
            return "get_sector_daily_tool", dict(sector_name=sector, trade_date=day)
        return None

    def market_context(self, name, args):
        count = args.get("lookback_days", 60 if name == "get_index_data" else 20)
        if type(count) is not int or not 1 <= count <= 365:
            raise ValueError("回看天数须为 1 至 365 的整数")
        native = None
        attempted = bool(self.bundle.get("online_enabled") and os.getenv("TUSHARE_TOKEN"))
        if attempted:
            token = research_snapshot.set({"as_of": self.bundle["as_of"]})
            try:
                report = self.native[name].invoke(dict(args, trade_date=self.day))
                # The native providers sometimes return a formatted heading even for zero rows.
                markers = {"get_index_data": r"\(\d{6}\.(?:SH|SZ)\):\s*[1-9]\d*",
                           "get_market_breadth": r"(?:A股上市公司|两市总成交额|总市值|总成交额|股票数量|上市公司总数)[^\n]{0,16}[1-9]",
                           "get_sector_daily_tool": r"收盘[:：]\s*[1-9]\d*"}
                if isinstance(report, str) and re.search(markers[name], report):
                    native = report
            except (ValueError, OSError, ImportError, RuntimeError, KeyError, TypeError):
                pass
            finally:
                research_snapshot.reset(token)
        native_sector = None
        if native and name == "get_sector_daily_tool":
            stamp = re.search(r"实际行情日:\s*(\d{4}-?\d{2}-?\d{2})", native)
            latest = date.fromisoformat(stamp[1]).isoformat() if stamp else None
            if latest and self.bundle["start"] <= latest <= self.day:
                native_sector = dict(source="TradingAgents-CN / Tushare", classification="同花顺行业指数",
                                     requested_date=self.day, latest_date=latest, provider_report=native,
                                     status="截至查询日" if latest == self.day else "仅有较早数据")
                if latest == self.day:
                    return native_sector
            else:
                native = None
        if name == "get_index_data":
            result = index_context(self.bundle, count)
        elif name == "get_market_breadth":
            result = breadth_context(self.bundle)
            if native:
                return dict(source="TradingAgents-CN / Tushare + qutumn", requested_date=self.day,
                            market_statistics=native, csi300_ma20_breadth=result)
        else:
            result = sector_context(self.bundle, args["sector_name"], count, os.getenv("ETF_SECTOR_CACHE"))
        if native_sector and native_sector["latest_date"] >= result.get("latest_date", ""):
            return native_sector
        if native:
            if name == "get_index_data" and not result["indexes"]:
                return dict(source="TradingAgents-CN / Tushare", requested_date=self.day, provider_report=native,
                            supplemental_gaps=result["gaps"])
            result["native_report"] = native
            result["native_source"] = "TradingAgents-CN / Tushare"
        elif attempted:
            result["native_gap"] = "原版工具未返回可核验的市场数据"
        return result

    def news(self, ticker):
        if ticker in self.news_cache:
            return self.news_cache[ticker]
        items = []
        for row in self.bundle["news"]:
            stamp = datetime.fromisoformat(row["published"])
            if stamp.tzinfo is None or stamp > self.cutoff:
                continue
            text = " ".join(str(row.get(k, "")) for k in ("title", "excerpt", "stock", "code", "ticker"))
            if ticker in self.holdings and not any(key and key in text for key in (ticker, self.holdings[ticker])):
                continue
            items.append(dict(title=row["title"], publish_time=stamp.astimezone(self.cutoff.tzinfo).replace(tzinfo=None),
                              summary=row.get("excerpt", ""), url=row.get("url", ""), source=row.get("source", ""),
                              data_source=row.get("source") or "archive"))
        if self.bundle.get("online_enabled") and ticker in self.holdings:
            items.extend(self.fetch_news(ticker, datetime.fromisoformat(self.bundle["start"]),
                                         self.cutoff.replace(tzinfo=None), limit=20))
        self.news_cache[ticker] = items
        return items

    def invoke(self, role, name, args):
        if name not in self.names(role) or not isinstance(args, dict):
            raise ValueError("该角色没有申请的工具")
        if name == "get_research_feed":
            return feed_result(self.bundle, args)
        if name in DESCRIPTIONS:
            query = args["query"]
            if not isinstance(query, str) or len(query) > 128:
                raise ValueError("无效查询")
            return lookup(self.bundle, name.removeprefix("lookup_"), query)
        ticker = args.get("ticker", self.bundle["ticker"])
        if name in FILINGS and ticker not in self.holdings:
            raise ValueError("公司财报工具需要已披露持仓的股票代码")
        if ticker != self.bundle["ticker"] and (name not in (*NEWS, *FILINGS) or ticker not in self.holdings):
            raise ValueError("代码超出本次ETF或已披露持仓范围")
        for k in ("curr_date", "end_date", "trade_date"):
            if args.get(k) and args[k] != self.day:
                raise ValueError("工具分析日期必须与冻结日期一致")
        if name in MARKET_CONTEXT_TOOLS:
            return self.market_context(name, args)
        snapshot = {"start": self.bundle["start"]}
        args = dict(args, ticker=ticker)
        if name in FILINGS:
            filing = max((r for r in self.bundle["financials"] if r["code"] == ticker and r["published_date"] < self.day),
                         key=lambda r: (r["period_end"], r["published_date"]), default=None)
            if not filing:
                return {"gap": "截止时没有已披露的成分股财报"}
            figures = filing["figures"]
            data = {"report_period": filing["period_end"], "filing_source":
                    f"公告 {filing['report_url']}，披露日 {filing['published_date']}"}
            for key, target in (("revenue", "revenue"), ("operating_cash_flow", "n_cashflow_act")):
                data[target] = (figures.get(key) or {}).get("current_cny")
            profit = figures.get("net_profit") or {}
            # The original tool labels this item as attributable to the parent.
            if "归属于母公司" in str(profit.get("source_row", "")):
                data["net_profit"] = profit.get("current_cny")
            snapshot["filing"] = data
        elif name in NEWS:
            if not 1 <= args.get("limit", 12) <= 60:
                raise ValueError("最多取60条标题")
            snapshot["news"] = self.news(ticker)
            args["curr_date"] = self.day
        else:
            dates = sorted((d, v) for d, v in price_series(self.bundle).items()
                           if self.bundle["start"] <= d <= self.day)
            history = [dict(date=d, close=v) for d, v in dates]
            if name == "get_etf_fundamentals":
                nav = max((r for r in self.bundle.get("nav", []) if r["date"] <= self.day),
                          key=lambda r: r["date"], default={})
                quotes = {r["date"]: r for r in self.bundle["prices"]}
                close = dict(dates).get(self.day)
                today = quotes.get(self.day, {})
                index = next((r["indexCode"] for r in reversed(self.bundle["tracking_index"])
                              if r["date"] <= self.day and r.get("indexCode")), None)
                spot = dict(code=ticker, full_symbol=ticker + ".SH", track_index=index,
                            latest_price=close, market_price=close, change_pct=today.get("return_pct"),
                            amount=today.get("amount_cny"), unit_nav=nav.get("unit_nav"), latest_nav_date=nav.get("date"))
                if close and nav.get("date") == self.day and float(nav.get("unit_nav") or 0) > 0:
                    spot["premium_discount_rate"] = round(100 * (close / float(nav["unit_nav"]) - 1), 4)
                for row in history:
                    row["pct_chg"] = quotes.get(row["date"], {}).get("return_pct")
                snapshot.update(spot=spot, history=history[::-1])
                args["curr_date"] = self.day
            else:
                start = max(args.get("start_date") or self.bundle["start"], self.bundle["start"])
                snapshot["history"] = [row for row in history if row["date"] >= start]
                args.update(start_date=start, end_date=self.day)
                requested = [i.strip() for i in args["indicators"].lower().split(",")]
                if not requested or any(i not in WINDOWS for i in requested):
                    raise ValueError("可用收盘价指标：" + ",".join(WINDOWS))
                breaks = price_breaks(dict(dates), self.bundle)
                if breaks:
                    snapshot["history"] = [r for r in snapshot["history"] if r["date"] >= breaks[-1]]
                count = len(snapshot["history"])
                available = [i for i in requested if count >= WINDOWS[i]]
                missing = [i for i in requested if i not in available]
                if not available:
                    return {"gap": "同一价格口径的历史长度不足", "requested": requested, "observations": count}
                args.update(indicators=",".join(available), start_date=snapshot["history"][0]["date"])
        token = research_snapshot.set(snapshot)
        try:
            result = self.native[name].invoke(args)
            if name == "get_technical_indicators" and missing:
                return {"report": result, "unavailable_indicators": missing, "observations": count}
            return result
        finally:
            research_snapshot.reset(token)
