"""Research and risk turn routing shared by graph and local runners."""
import logging

logger = logging.getLogger(__name__)


class DebateRouting:
    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_debate(self, state: dict) -> str:
        """Determine if debate should continue."""
        current_count = state["investment_debate_state"]["count"]
        max_count = 2 * self.max_debate_rounds
        current_speaker = state["investment_debate_state"]["current_response"]

        # 🔍 详细日志
        logger.info(f"🔍 [投资辩论控制] 当前发言次数: {current_count}, 最大次数: {max_count} (配置轮次: {self.max_debate_rounds})")
        logger.info(f"🔍 [投资辩论控制] 当前发言者: {current_speaker}")

        if current_count >= max_count:
            logger.info(f"✅ [投资辩论控制] 达到最大次数，结束辩论 -> Research Manager")
            return "Research Manager"

        next_speaker = "Bear Researcher" if current_speaker.startswith("Bull") else "Bull Researcher"
        logger.info(f"🔄 [投资辩论控制] 继续辩论 -> {next_speaker}")
        return next_speaker

    def should_continue_risk_analysis(self, state: dict) -> str:
        """Determine if risk analysis should continue."""
        current_count = state["risk_debate_state"]["count"]
        max_count = 3 * self.max_risk_discuss_rounds
        latest_speaker = state["risk_debate_state"]["latest_speaker"]

        # 🔍 详细日志
        logger.info(f"🔍 [风险讨论控制] 当前发言次数: {current_count}, 最大次数: {max_count} (配置轮次: {self.max_risk_discuss_rounds})")
        logger.info(f"🔍 [风险讨论控制] 最后发言者: {latest_speaker}")

        if current_count >= max_count:
            logger.info(f"✅ [风险讨论控制] 达到最大次数，结束讨论 -> Risk Judge")
            return "Risk Judge"

        # 确定下一个发言者
        if latest_speaker.startswith("Risky"):
            next_speaker = "Safe Analyst"
        elif latest_speaker.startswith("Safe"):
            next_speaker = "Neutral Analyst"
        else:
            next_speaker = "Risky Analyst"

        logger.info(f"🔄 [风险讨论控制] 继续讨论 -> {next_speaker}")
        return next_speaker
