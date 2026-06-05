"""
VoltOptimizer - Logging Infrastructure
=======================================
Logging system that displays terminal output in a colourised,
structured format. Agent reasoning processes are logged here.
"""

import sys
import datetime
from colorama import init, Fore, Style

init(autoreset=True)  # Initialise colorama for Windows compatibility


class VoltLogger:
    """
    Colourised, structured terminal log system.
    Assigns a distinct prefix and colour to each agent and module.
    """

    COLORS = {
        "system": Fore.CYAN,
        "dl_model": Fore.BLUE,
        "battery_guardian": Fore.RED,
        "grid_tariff": Fore.YELLOW,
        "smart_trip": Fore.GREEN,
        "orchestrator": Fore.MAGENTA,
        "tool": Fore.WHITE,
        "reasoning": Fore.LIGHTCYAN_EX,
        "warning": Fore.LIGHTYELLOW_EX,
        "error": Fore.LIGHTRED_EX,
        "success": Fore.LIGHTGREEN_EX,
    }

    ICONS = {
        "system": "⚡",
        "dl_model": "🧠",
        "battery_guardian": "🛡️",
        "grid_tariff": "💰",
        "smart_trip": "🗺️",
        "orchestrator": "🎯",
        "tool": "🔧",
        "reasoning": "💭",
        "warning": "⚠️",
        "error": "❌",
        "success": "✅",
    }

    def __init__(self):
        self.log_history = []

    def _timestamp(self) -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    def log(self, category: str, message: str, indent: int = 0):
        """Main log function."""
        color = self.COLORS.get(category, Fore.WHITE)
        icon = self.ICONS.get(category, "📌")
        prefix = "  " * indent
        timestamp = self._timestamp()

        formatted = (
            f"{Style.DIM}[{timestamp}]{Style.RESET_ALL} "
            f"{color}{icon} [{category.upper()}]{Style.RESET_ALL} "
            f"{prefix}{color}{message}{Style.RESET_ALL}"
        )

        print(formatted)
        self.log_history.append({
            "time": timestamp,
            "category": category,
            "message": message,
        })

    def separator(self, char: str = "═", length: int = 70):
        """Visual separator line."""
        print(f"\n{Fore.CYAN}{char * length}{Style.RESET_ALL}\n")

    def header(self, title: str):
        """Large section header."""
        self.separator()
        padding = (68 - len(title)) // 2
        print(f"{Fore.CYAN}║{' ' * padding}{Style.BRIGHT}{title}"
              f"{' ' * padding}║{Style.RESET_ALL}")
        self.separator()

    def agent_thinking(self, agent_name: str, thought: str, step: int = 0):
        """Logs an agent's reasoning process."""
        self.log("reasoning",
                 f"[{agent_name}] Step {step}: {thought}", indent=1)

    def agent_action(self, agent_name: str, action: str):
        """Logs an agent action."""
        self.log("tool", f"[{agent_name}] Action: {action}", indent=1)

    def agent_result(self, agent_name: str, result: str):
        """Logs an agent result."""
        self.log("success", f"[{agent_name}] Result: {result}", indent=1)

    def warning(self, message: str):
        self.log("warning", message)

    def error(self, message: str):
        self.log("error", message)


# Global logger instance
logger = VoltLogger()
