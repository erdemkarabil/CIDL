"""
VoltOptimizer - Temel Ajan Sınıfı (Base Agent)
================================================
Tüm ajanların miras aldığı temel sınıf. CrewAI benzeri bir mimari ile
her ajan:
  - Bir rol ve göreve sahiptir
  - Tool'lar kullanabilir
  - Akıl yürütme (reasoning) yapabilir
  - Kendi kendini düzeltebilir (self-correction)
  - Diğer ajanlarla mesaj alışverişi yapabilir

Mimari Deseni: ReAct (Reasoning + Acting)
    1. Gözlemle → 2. Düşün → 3. Eylem Yap → 4. Değerlendir → 5. Tekrarla
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AGENT_CONFIG
from utils.logger import logger


class BaseAgent:
    """
    Tüm VoltOptimizer ajanlarının temel sınıfı.

    CrewAI framework'üne benzer bir yapı sunar:
      - role: Ajanın rolü (örn: "Batarya Sağlık Koruyucusu")
      - goal: Ajanın amacı
      - backstory: Ajanın arka plan hikayesi
      - tools: Kullanabileceği araçlar listesi
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: list = None,
        verbose: bool = None,
        max_reasoning_steps: int = None,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = (verbose if verbose is not None
                        else AGENT_CONFIG["verbose"])
        self.max_reasoning_steps = (
            max_reasoning_steps or AGENT_CONFIG["max_reasoning_steps"]
        )

        # Akıl yürütme geçmişi
        self.reasoning_history = []
        # Mesaj kutusu (diğer ajanlardan gelen mesajlar)
        self.inbox = []
        # Üretilen çıktılar
        self.outputs = {}

        if self.verbose:
            logger.log("system",
                       f"Ajan oluşturuldu: {self.role}")

    def use_tool(self, tool_name: str, tool_instance, method: str,
                 **kwargs) -> dict:
        """
        Bir aracı kullanır ve sonucunu döndürür.

        Args:
            tool_name: Aracın görünen adı
            tool_instance: Araç nesnesi
            method: Çağrılacak metod adı
            **kwargs: Metoda geçirilecek parametreler

        Returns:
            Araç çıktısı
        """
        if self.verbose:
            logger.agent_action(self.role,
                                f"'{tool_name}' aracı kullanılıyor → "
                                f"{method}()")

        tool_method = getattr(tool_instance, method)
        result = tool_method(**kwargs)

        if self.verbose:
            logger.agent_action(self.role,
                                f"'{tool_name}' aracından sonuç alındı ✓")

        return result

    def reason(self, observation: dict, step: int = 0) -> str:
        """
        Gözleme dayalı akıl yürütme.

        Alt sınıflar tarafından override edilir.

        Args:
            observation: Gözlem verisi
            step: Akıl yürütme adımı

        Returns:
            Akıl yürütme metni
        """
        thought = f"Gözlem alındı, analiz ediliyor..."
        self.reasoning_history.append({
            "step": step,
            "observation": observation,
            "thought": thought,
        })

        if self.verbose:
            logger.agent_thinking(self.role, thought, step)

        return thought

    def self_correct(self, previous_output: dict,
                     feedback: str) -> dict:
        """
        Kendi kendini düzeltme mekanizması.

        Önceki çıktıyı ve geri bildirimi değerlendirerek
        düzeltilmiş bir çıktı üretir.

        Args:
            previous_output: Önceki çıktı
            feedback: Geri bildirim

        Returns:
            Düzeltilmiş çıktı
        """
        if self.verbose:
            logger.agent_thinking(
                self.role,
                f"Kendini düzeltme aktif: {feedback}"
            )

        corrected = previous_output.copy()
        corrected["self_corrected"] = True
        corrected["correction_reason"] = feedback

        return corrected

    def receive_message(self, from_agent: str, message: dict):
        """
        Başka bir ajandan mesaj alır.

        Args:
            from_agent: Gönderen ajan adı
            message: Mesaj içeriği
        """
        self.inbox.append({
            "from": from_agent,
            "message": message,
        })

        if self.verbose:
            logger.log("system",
                       f"[{self.role}] ← Mesaj alındı: [{from_agent}]")

    def send_message(self, to_agent, message: dict):
        """
        Başka bir ajana mesaj gönderir.

        Args:
            to_agent: Hedef ajan nesnesi
            message: Mesaj içeriği
        """
        if self.verbose:
            logger.log("system",
                       f"[{self.role}] → Mesaj gönderiliyor: [{to_agent.role}]")

        to_agent.receive_message(self.role, message)

    def execute(self, task_input: dict) -> dict:
        """
        Ana görev yürütme fonksiyonu.
        Alt sınıflar tarafından override edilir.

        Args:
            task_input: Görev girdisi

        Returns:
            Görev çıktısı
        """
        raise NotImplementedError(
            "execute() metodu alt sınıfta tanımlanmalıdır"
        )

    def get_summary(self) -> dict:
        """Ajan özetini döndürür."""
        return {
            "role": self.role,
            "goal": self.goal,
            "tools": [type(t).__name__ for t in self.tools],
            "reasoning_steps": len(self.reasoning_history),
            "messages_received": len(self.inbox),
            "outputs": list(self.outputs.keys()),
        }
