"""Role-scoped native tool calls backed by the host experiment's dated catalogue."""
import json
import sys

from cli.etf_graph_tools import ResearchTools, ROLE_TOOLS


class QuantResearchTools:
    def __init__(self, brief, output):
        self.output = output
        self.catalog = brief["tools"]
        self.funds = brief.get("funds", [])
        self.day = brief["cutoff"][:10]
        self.schemas = {}
        for name, spec in self.catalog.items():
            spec = spec if isinstance(spec, dict) else {"description": spec}
            arguments = spec.get("arguments")
            params = dict(arguments) if isinstance(arguments, dict) else {"type": "object", "properties": {}}
            props = dict(params.get("properties", {}))
            if spec.get("fund"):
                props["fund"] = {"type": "string", "enum": self.funds}
                if len(self.funds) > 1:
                    params["required"] = [*params.get("required", []), "fund"]
            if name in ("training_bars", "auction_outcomes"):
                props.update({k: {"type": "string"} for k in ("symbol", "start", "end")})
                if name == "training_bars":
                    props["fields"] = {"type":"array", "items":{"type":"string", "enum":spec["fields"]}}
                    params["required"] = ["fields"]
            self.schemas[name] = {"type":"function", "function":{"name":name,
                "description": str(spec.get("description") or arguments or name)[:900],
                "parameters": dict(params, properties=props)}}

    def names(self, role):
        common = ["research_universe", "archived_reports", "training_bars", "development_market"]
        if role in ("research_manager_v2", "trader_v2", "risk_manager_v2", "bull", "bear", "safe_analyst_v2", "risky", "neutral"):
            return list(self.schemas)
        names = [*ROLE_TOOLS.get(role, ()), *common]
        if role in ("market_analyst_v2", "index"):
            names.append("auction_outcomes")
        if role in ("financials", "fundamentals_analyst_v2", "news_analyst_v2", "sector", "sentiment"):
            names.extend(["stock_disclosures", "stock_news", "stock_financials"])
        return [name for name in dict.fromkeys(names) if name in self.schemas]

    def preflight(self, role):
        if len(self.funds) == 1:
            query = ResearchTools.default_query(role, self.funds[0], self.day)
            if query and query[0] in self.catalog:
                return query[0], dict(query[1], fund=self.funds[0])
        if role == "market_analyst_v2" and "training_bars" in self.catalog:
            return "training_bars", {"fields":["close"]}
        if len(self.funds) == 1 and role in ("etf_analyst_v2", "news_analyst_v2"):
            return ("lookup_market" if role == "etf_analyst_v2" else "lookup_news",
                    {"query":"", "fund":self.funds[0]})
        if role == "index" and "research_universe" in self.catalog:
            return "research_universe", {}
        return None

    def emit(self, kind, **fields):
        print(json.dumps(dict(type=kind, **fields), ensure_ascii=False), file=self.output, flush=True)

    @staticmethod
    def excerpt(value):
        text = json.dumps(value, ensure_ascii=False)
        if len(text) <= 4200:
            return value
        return {"excerpt": text[:4000], "truncated":True}

    def invoke(self, role, name, args):
        if name not in self.names(role):
            raise ValueError("工具不在本角色的研究范围内")
        self.emit("query", role=role, request={"tool":name, "arguments":args})
        return json.loads(sys.stdin.readline())
