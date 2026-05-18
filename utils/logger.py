"""
VoltOptimizer - Loglama Altyapısı
==================================
Terminal çıktılarını renklendirilmiş ve yapılandırılmış formatta gösteren
loglama sistemi. Ajanların akıl yürütme süreçleri burada loglanır.
"""

import sys
import datetime
from colorama import init, Fore, Style

init(autoreset=True)  # Windows uyumluluğu için colorama başlat


class VoltLogger:
    """
    Renkli ve yapılandırılmış terminal log sistemi.
    Her ajan ve modül için ayrı prefix ve renk ataması yapar.
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
        """Ana log fonksiyonu."""
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
        """Görsel ayırıcı satır."""
        print(f"\n{Fore.CYAN}{char * length}{Style.RESET_ALL}\n")

    def header(self, title: str):
        """Büyük başlık."""
        self.separator()
        padding = (68 - len(title)) // 2
        print(f"{Fore.CYAN}║{' ' * padding}{Style.BRIGHT}{title}"
              f"{' ' * padding}║{Style.RESET_ALL}")
        self.separator()

    def agent_thinking(self, agent_name: str, thought: str, step: int = 0):
        """Ajan akıl yürütme süreci logu."""
        self.log("reasoning",
                 f"[{agent_name}] Adım {step}: {thought}", indent=1)

    def agent_action(self, agent_name: str, action: str):
        """Ajan eylem logu."""
        self.log("tool", f"[{agent_name}] Eylem: {action}", indent=1)

    def agent_result(self, agent_name: str, result: str):
        """Ajan sonuç logu."""
        self.log("success", f"[{agent_name}] Sonuç: {result}", indent=1)

    def warning(self, message: str):
        self.log("warning", message)

    def error(self, message: str):
        self.log("error", message)


# Global logger instance
logger = VoltLogger()
