"""
RAG-ассистент для анализа донесений разведки
Версия: 1.0 (On-premise, изолированная сеть)
"""

import json
import re
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ
# ==========================================

@dataclass
class IntelReport:
  """Разведывательное донесение"""
  id: str
  timestamp: str
  coordinates: Tuple[float, float]
  location: str
  enemy_forces: Dict[str, int]
  equipment: Dict[str, int]
  activity: str
  confidence: str  # высокая/средняя/низкая
  source: str
  raw_text: str
  
  def to_dict(self):
      return asdict(self)


class IntelDataGenerator:
  """Генератор синтетических разведывательных донесений"""
  
  def __init__(self):
      self.reports = []
      
  def generate_reports(self, count: int = 10) -> List[IntelReport]:
      """Генерация донесений"""
      templates = [
          {
              "coords": (48.8566, 37.6173),
              "location": "северная окраина н.п. Красное",
              "forces": {"пехота": 15, "саперы": 3},
              "equip": {"танк": 3, "БТР": 2, "грузовик": 4},
              "activity": "инженерное оборудование позиций",
              "conf": "высокая"
          },
          {
              "coords": (48.7234, 37.8123),
              "location": "высота 215.3, лесной массив",
              "forces": {"пехота": 8},
              "equip": {"БМП": 4, "БТР": 1},
              "activity": "засада на дороге подхода",
              "conf": "средняя"
          },
          {
              "coords": (48.9123, 37.4567),
              "location": "южная промзона",
              "forces": {"пехота": 25, "снайперы": 2},
              "equip": {"танк": 5, "БТР": 6, "САУ": 2},
              "activity": "сосредоточение перед наступлением",
              "conf": "высокая"
          },
          {
              "coords": (48.6543, 37.7890),
              "location": "переправа через р. Северский Донец",
              "forces": {"инженеры": 10},
              "equip": {"понтон": 2, "грузовик": 8, "БТР": 3},
              "activity": "оборудование переправы",
              "conf": "высокая"
          },
          {
              "coords": (48.8123, 37.2345),
              "location": "ж/д станция 'Мирная'",
              "forces": {"охрана": 12},
              "equip": {"вагон": 15, "тепловоз": 2},
              "activity": "погрузка техники на платформы",
              "conf": "средняя"
          },
          {
              "coords": (48.7345, 37.6789),
              "location": "база отдыха 'Лесная'",
              "forces": {"штаб": 8, "связисты": 4},
              "equip": {"КШМ": 3, "радиостанция": 5},
              "activity": "развертывание пунктов управления",
              "conf": "высокая"
          },
          {
              "coords": (48.8901, 37.8901),
              "location": "траса М-03, км 45",
              "forces": {"пехота": 6},
              "equip": {"танк": 2, "БТР": 2},
              "activity": "передвижение колонны на запад",
              "conf": "низкая"
          },
          {
              "coords": (48.5678, 37.4567),
              "location": "н.п. Степное, школа №3",
              "forces": {"пехота": 30, "минометчики": 6},
              "equip": {"танк": 4, "БТР": 5, "миномет": 6},
              "activity": "занятие опорного пункта",
              "conf": "высокая"
          },
          {
              "coords": (48.6789, 37.7890),
              "location": "склад ГСМ, северная промзона",
              "forces": {"охрана": 10, "пожарные": 4},
              "equip": {"цистерна": 12, "грузовик": 6},
              "activity": "перекачка топлива",
              "conf": "средняя"
          },
          {
              "coords": (48.7890, 37.1234),
              "location": "аэродром временного базирования",
              "forces": {"техники": 15, "охрана": 20},
              "equip": {"вертолет": 4, "ЗРК": 2, "радар": 1},
              "activity": "обслуживание авиации",
              "conf": "высокая"
          }
      ]
      
      for i, t in enumerate(templates[:count], 1):
          raw_text = self._format_report_text(t, i)
          report = IntelReport(
              id=f"RPT-{datetime.datetime.now().year}-{i:03d}",
              timestamp=datetime.datetime.now().isoformat(),
              coordinates=t["coords"],
              location=t["location"],
              enemy_forces=t["forces"],
              equipment=t["equip"],
              activity=t["activity"],
              confidence=t["conf"],
              source="БПЛА 'Орлан-10'" if i % 2 == 0 else "Агентурная разведка",
              raw_text=raw_text
          )
          self.reports.append(report)
          
      return self.reports
  
  def _format_report_text(self, data: dict, num: int) -> str:
      """Форматирование текста донесения"""
      forces_str = ", ".join([f"{k}: {v} чел" for k, v in data["forces"].items()])
      equip_str = ", ".join([f"{k}: {v} ед" for k, v in data["equip"].items()])
      
      return f"""
ДОНЕСЕНИЕ РАЗВЕДКИ №{num:03d}
Время: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}
Координаты: {data['coords'][0]:.4f} N, {data['coords'][1]:.4f} E
Район: {data['location']}

ОБНАРУЖЕННЫЕ СИЛЫ:
Живая сила: {forces_str}
Техника: {equip_str}

ДЕЯТЕЛЬНОСТЬ: {data['activity']}

ДОВЕРИЕ: {data['conf'].upper()}
ИСТОЧНИК: {'БПЛА' if num % 2 == 0 else 'Агентурная разведка'}
      """.strip()


# ==========================================
# 2. БАЗА ЗНАНИЙ (УСТАВЫ И ПОСОБИЯ)
# ==========================================

class TacticalKnowledgeBase:
  """Локальная база знаний (20 страниц эмуляция)"""
  
  def __init__(self):
      self.documents = []
      self._load_knowledge()
      
  def _load_knowledge(self):
      """Загрузка тактических документов (только проверенные источники)"""
      
      knowledge_docs = [
          {
              "title": "Устав РВиА. Тактика танковых подразделений",
              "content": """
При обнаружении танковой роты (3-4 танка) противник, вероятно, готовит:
- Наступление на узком участке фронта
- Контратаку по выявленным позициям
- Оборону ключевого рубежа

Рекомендуется: усиление разведки, готовность к маневру резервов.
              """,
              "classification": "ДСП",
              "source": "Устав РВиА, изд. 2023"
          },
          {
              "title": "Пособие по противодиверсионной борьбе",
              "content": """
Обнаружение саперных групп указывает на:
1. Подготовку проходов в минных полях
2. Установку минно-взрывных заграждений
3. Инженерную разведку местности

Требуется: усиление наблюдения, проверка маршрутов.
              """,
              "classification": "ДСП",
              "source": "ПУ-34, п. 45-67"
          },
          {
              "title": "Тактика применения БПЛА 'Орлан-10'",
              "content": """
Разведка с БПЛА обеспечивает:
- Выявление позиций ПВО (визуальный/тепловизионный контакт)
- Корректировку огня артиллерии
- Оценку результатов ударов

Ограничения: погодные условия, радиоэлектронная борьба противника.
              """,
              "classification": "ДСП",
              "source": "Руководство по эксплуатации БПЛА"
          },
          {
              "title": "Анализ сосредоточения войск",
              "content": """
Признаки подготовки наступления:
- Накопление техники в 15-20 км от линии соприкосновения
- Развертывание пунктов управления
- Инженерное оборудование позиций артиллерии
- Переправы через водные преграды

Время готовности: 24-48 часов с момента завершения сосредоточения.
              """,
              "classification": "Секретно",
              "source": "Аналитическая записка ГРУ №45/2023"
          },
          {
              "title": "Противодействие засадам",
              "content": """
Признаки засады в лесном массиве:
- Отсутствие видимого перемещения при наличии техники
- Свежие следы гусеничной техники, ведущие в лес
- Нарушение целостности растительности

Меры: обход, зачистка с применением тепловизоров, огневое прикрытие.
              """,
              "classification": "ДСП",
              "source": "Наставление по тактике, ч. 3"
          },
          {
              "title": "Оценка инженерной подготовки",
              "content": """
Оборудование переправ указывает на:
- Намерение форсировать водную преграду
- Подготовку к отступлению
- Создание резервных маршрутов

Критичность: переправы — уязвимое звено, требуют прикрытия ПВО.
              """,
              "classification": "ДСП",
              "source": "Инженерный устав ВС РФ"
          },
          {
              "title": "Анализ железнодорожных перевозок",
              "content": """
Погрузка техники на ж/д платформы:
- Массовое переброшение резервов
- Эвакуация поврежденной техники
- Перегруппировка на другой участок

Скорость: до 500 км/сутки, сложно перехватить.
              """,
              "classification": "ДСП",
              "source": "Справочник военного железнодорожника"
          },
          {
              "title": "Разведка пунктов управления",
              "content": """
Признаки штаба батальона/полка:
- Концентрация КШМ (командно-штабных машин)
- Антенны радиостанций различных диапазонов
- Усиленная охрана (в 2-3 раза больше обычного)

Приоритет цели: высокий (нарушение управления дезорганизует противника).
              """,
              "classification": "Секретно",
              "source": "Методичка ГРУ по целеуказанию"
          },
          {
              "title": "Оценка авиационной активности",
              "content": """
Вертолеты на временном аэродроме:
- Ударные (Ка-52, Ми-28): готовность к поддержке наступления
- Транспортные (Ми-8): переброска десанта
- Разведывательные: уточнение целей

Дальность действия: до 300 км от линии фронта.
              """,
              "classification": "ДСП",
              "source": "Справочник ВВС и ПВО"
          },
          {
              "title": "Анализ складов ГСМ",
              "content": """
Склады горюче-смазочных материалов:
- Обеспечивают операции на 7-10 суток
- Критичная уязвимость (взрывоопасность)
- Требуют постоянной охраны

Уничтожение: парализует механизированные части противника.
              """,
              "classification": "ДСП",
              "source": "Тыловой устав ВС РФ"
          },
          {
              "title": "Противодействие снайперам",
              "content": """
Обнаружение снайперских пар:
- Работа в пригородной застройке или высотных зданиях
- Действуют на дистанции 300-800 м
- Приоритетные цели: офицеры, пулеметчики, операторы ПТРК

Меры: дымовые завесы, бронежилеты, контрснайперские группы.
              """,
              "classification": "ДСП",
              "source": "Наставление по городским действиям"
          },
          {
              "title": "Оценка минометных позиций",
              "content": """
Минометы (82-мм, 120-мм) в опорном пункте:
- Дальность: 4-7 км
- Быстрое развертывание (2-3 минуты)
- Массированный залп по площадям

Требуется: контрбатарейная борьба или подавление РЭБ.
              """,
              "classification": "ДСП",
              "source": "Артиллерийский устав"
          },
          {
              "title": "Тактика противодействия БПЛА",
              "content": """
Обнаружение станций РЭБ и радаров:
- Противник усиливает ПВО
- Возможно применение средств радиоэлектронной борьбы
- Ограничение эффективности нашей разведки

Меры: смена частот, использование кабельных линий, ложные цели.
              """,
              "classification": "Секретно",
              "source": "Инструкция РЭБ ВКС"
          },
          {
              "title": "Анализ передвижения колонн",
              "content": """
Перемещение танков и БТР по трассам:
- Дневное движение = спешка или отсутствие угрозы с воздуха
- Ночное = попытка скрыть перегруппировку
- Колонна без прикрытия ПВО = уязвимая цель

Рекомендуется: удар авиации или ПТРК.
              """,
              "classification": "ДСП",
              "source": "Наставление по тактике"
          },
          {
              "title": "Оценка опорных пунктов",
              "content": """
Занятие школ, больниц, админзданий:
- Использование гражданских объектов как щита
- Укрепленные позиции внутри зданий
- Сложность штурма (риск гражданских жертв)

Требуется: точечное оружие, переговоры об эвакуации мирных.
              """,
              "classification": "ДСП",
              "source": "Правила ведения боевых действий"
          },
          {
              "title": "Прогнозирование действий противника",
              "content": """
Шаблоны поведения при обороне:
1. Подготовка позиций (3-5 суток)
2. Выставление наблюдателей и снайперов
3. Минирование подходов
4. Размещение резервов во втором эшелоне

Прорыв: требует превосходства 3:1 по силам.
              """,
              "classification": "Секретно",
              "source": "Аналитический центр ГШ ВС РФ"
          },
          {
              "title": "Координаты и ориентиры",
              "content": """
Система координат WGS-84:
- Точность БПЛА 'Орлан-10': ±50 м
- Точность агентурной разведки: зависит от источника
- Уточнение: по крупным ориентирам (перекрестки, здания, мосты)

Важно: перепроверка несколькими источниками.
              """,
              "classification": "ДСП",
              "source": "Инструкция топографической службы"
          },
          {
              "title": "Оценка доверия к данным",
              "content": """
Уровни достоверности:
- ВЫСОКАЯ: визуальный контакт, фото/видео подтверждение
- СРЕДНЯЯ: данные агентуры, косвенные признаки
- НИЗКАЯ: слухи, неподтвержденные сообщения

Действия при низкой достоверности: усиление разведки, не принимать к сведению без подтверждения.
              """,
              "classification": "ДСП",
              "source": "Руководство разведчика"
          },
          {
              "title": "Взаимодействие родов войск",
              "content": """
Типичные сочетания:
- Танки + пехота = наступление
- БТР + саперы = инженерная разведка
- Артиллерия + разведка = корректировка огня

Анализ сочетаний позволяет предсказать намерения противника.
              """,
              "classification": "ДСП",
              "source": "Устав по взаимодействию"
          },
          {
              "title": "Ограничения разведки",
              "content": """
Что НЕ может дать разведка:
- Точные планы противника (только предположения)
- Намерения командования (только косвенные признаки)
- Моральное состояние личного состава

Выводы: требуют критического анализа, не являются приказами.
              """,
              "classification": "ДСП",
              "source": "Методические рекомендации ГРУ"
          }
      ]
      
      for doc in knowledge_docs:
          doc["id"] = hashlib.md5(doc["title"].encode()).hexdigest()[:8]
          self.documents.append(doc)
          
  def search(self, query: str, top_k: int = 3) -> List[Dict]:
      """Простой поиск по ключевым словам (вместо векторного)"""
      query_lower = query.lower()
      scores = []
      
      for doc in self.documents:
          score = 0
          content_lower = doc["content"].lower()
          
          # Подсчёт вхождений ключевых слов
          query_words = set(query_lower.split())
          for word in query_words:
              if len(word) > 3:  # Игнорируем короткие слова
                  score += content_lower.count(word)
          
          # Бонус за точные фразы
          if query_lower in content_lower:
              score += 10
              
          if score > 0:
              scores.append((score, doc))
              
      # Сортировка по релевантности
      scores.sort(reverse=True, key=lambda x: x[0])
      return [doc for _, doc in scores[:top_k]]


# ==========================================
# 3. СИСТЕМА КОНТРОЛЯ ГЕНЕРАЦИИ (БЕЗОПАСНОСТЬ)
# ==========================================

class SafetyFilter:
  """
  Жесткие правила фильтрации запрещающие прямые боевые приказы
  """
  
  # Запрещённые паттерны (регистронезависимо)
  FORBIDDEN_PATTERNS = [
      r'\b(атаковать|штурмовать|уничтожить|ликвидировать)\s+(сейчас|немедленно|срочно)\b',
      r'\b(открыть\s+огонь|стрелять|бить)\s+по\s+(координатам|цели)\b',
      r'\b(выслать|направить)\s+(удар|ракеты|авиацию)\b',
      r'\b(приказ\s*:|приказываю|велено)\b',
      r'\b(всем\s+подразделениям|всем\s+частям)\b',
  ]
  
  # Запрещённые категории ответов
  FORBIDDEN_CATEGORIES = [
      "прямой приказ к действию",
      "координаты цели для огня",
      "расписание операции",
      "пароли/коды доступа",
  ]
  
  # Обязательные дисклеймеры
  REQUIRED_DISCLAIMERS = [
      "Анализ носит рекомендательный характер",
      "Решение принимает командир",
      "Требуется дополнительная проверка",
  ]
  
  def __init__(self):
      self.violations_log = []
      
  def check_input(self, query: str) -> Tuple[bool, str]:
      """Проверка входного запроса"""
      query_lower = query.lower()
      
      # Проверка на попытку получить приказ
      if any(keyword in query_lower for keyword in ["прикажи", "приказ", "вели", "выполни"]):
          return False, "Запрос отклонён: система не даёт приказов, только анализ."
          
      # Проверка на запрос координат для огня
      if "координаты для огня" in query_lower or "целеуказание" in query_lower:
          return False, "Запрос отклонён: система не производит целеуказание."
          
      return True, "OK"
  
  def check_output(self, response: str, sources: List[Dict]) -> Tuple[bool, str]:
      """Проверка выходного ответа"""
      response_lower = response.lower()
      
      # Проверка запрещённых паттернов
      for pattern in self.FORBIDDEN_PATTERNS:
          if re.search(pattern, response_lower):
              self.violations_log.append({
                  "type": "forbidden_pattern",
                  "pattern": pattern,
                  "response": response[:100],
                  "timestamp": datetime.datetime.now().isoformat()
              })
              return False, "Ответ заблокирован: обнаружен запрещённый паттерн."
      
      # Проверка наличия источников
      if not sources:
          return False, "Ответ заблокирован: отсутствуют подтверждающие документы."
          
      return True, "OK"
  
  def add_disclaimers(self, response: str) -> str:
      """Добавление обязательных дисклеймеров"""
      disclaimer = "\n\n---\n⚠️ ВАЖНО:\n"
      for i, disc in enumerate(self.REQUIRED_DISCLAIMERS, 1):
          disclaimer += f"{i}. {disc}\n"
          
      return response + disclaimer


# ==========================================
# 4. RAG-АССИСТЕНТ (ЯДРО СИСТЕМЫ)
# ==========================================

class IntelRAGAssistant:
  """
  RAG-ассистент для анализа разведывательных донесений
  On-premise, без интернета, с полным аудитом
  """
  
  def __init__(self, audit_log_path: str = "audit_log.json"):
      self.knowledge_base = TacticalKnowledgeBase()
      self.safety_filter = SafetyFilter()
      self.audit_log_path = Path(audit_log_path)
      self.audit_log = []
      self.reports_db = []
      
      # Загрузка существующего аудита
      if self.audit_log_path.exists():
          with open(self.audit_log_path, 'r', encoding='utf-8') as f:
              self.audit_log = json.load(f)
              
  def load_reports(self, reports: List[IntelReport]):
      """Загрузка донесений в систему"""
      self.reports_db = reports
      
  def query(self, user_query: str, user_id: str = "anon") -> Dict:
      """
      Основной метод запроса к системе
      
      Возвращает:
      {
          "success": bool,
          "response": str,
          "sources": List[Dict],
          "related_reports": List[str],
          "safety_check": str,
          "timestamp": str
      }
      """
      timestamp = datetime.datetime.now().isoformat()
      request_id = hashlib.md5(f"{user_id}{timestamp}{user_query}".encode()).hexdigest()[:12]
      
      # 1. Проверка входного запроса
      input_ok, input_msg = self.safety_filter.check_input(user_query)
      if not input_ok:
          self._log_audit(request_id, user_id, user_query, None, "BLOCKED_INPUT", input_msg)
          return {
              "success": False,
              "response": input_msg,
              "sources": [],
              "related_reports": [],
              "safety_check": "BLOCKED",
              "timestamp": timestamp
          }
      
      # 2. Поиск в базе знаний
      relevant_docs = self.knowledge_base.search(user_query, top_k=3)
      
      # 3. Поиск связанных донесений
      related_reports = self._find_related_reports(user_query)
      
      # 4. Генерация ответа (упрощённая, без LLM — только on-premise)
      response = self._generate_response(user_query, relevant_docs, related_reports)
      
      # 5. Проверка выходного ответа
      output_ok, output_msg = self.safety_filter.check_output(response, relevant_docs)
      if not output_ok:
          self._log_audit(request_id, user_id, user_query, response, "BLOCKED_OUTPUT", output_msg)
          return {
              "success": False,
              "response": output_msg,
              "sources": [],
              "related_reports": [],
              "safety_check": "BLOCKED",
              "timestamp": timestamp
          }
      
      # 6. Добавление дисклеймеров
      response = self.safety_filter.add_disclaimers(response)
      
      # 7. Логирование успешного запроса
      self._log_audit(request_id, user_id, user_query, response, "SUCCESS", "OK", 
                     sources=[d["title"] for d in relevant_docs])
      
      return {
          "success": True,
          "response": response,
          "sources": relevant_docs,
          "related_reports": [r.id for r in related_reports],
          "safety_check": "PASSED",
          "timestamp": timestamp
      }
  
  def _find_related_reports(self, query: str) -> List[IntelReport]:
      """Поиск связанных донесений по ключевым словам"""
      query_lower = query.lower()
      related = []
      
      for report in self.reports_db:
          score = 0
          text = report.raw_text.lower()
          
          # Проверка упоминания техники
          for equip_type in report.equipment.keys():
              if equip_type in query_lower:
                  score += 5
                  
          # Проверка локации
          if report.location.lower() in query_lower:
              score += 3
              
          # Проверка активности
          if report.activity.lower() in query_lower:
              score += 2
              
          if score > 0:
              related.append((score, report))
              
      related.sort(reverse=True, key=lambda x: x[0])
      return [r for _, r in related[:3]]
  
  def _generate_response(self, query: str, docs: List[Dict], reports: List[IntelReport]) -> str:
      """
      Генерация ответа на основе найденных документов и донесений
      (Шаблонный метод вместо LLM для работы без интернета)
      """
      # Анализ запроса
      query_lower = query.lower()
      
      # Определение типа запроса
      if "что" in query_lower and ("означает" in query_lower or "указывает" in query_lower):
          return self._generate_analysis_response(docs, reports)
      elif "как" in query_lower and ("действовать" in query_lower or "реагировать" in query_lower):
          return self._generate_recommendation_response(docs, reports)
      elif "сводка" in query_lower or "обстановка" in query_lower:
          return self._generate_summary_response(reports)
      else:
          return self._generate_general_response(docs, reports)
  
  def _generate_analysis_response(self, docs: List[Dict], reports: List[IntelReport]) -> str:
      """Анализ обстановки"""
      response = "АНАЛИЗ РАЗВЕДЫВАТЕЛЬНОЙ ОБСТАНОВКИ\n\n"
      
      if reports:
          response += "На основе донесений:\n"
          for r in reports[:2]:
              response += f"- {r.id}: {r.location}, обнаружено {sum(r.equipment.values())} ед. техники ({r.confidence} достоверность)\n"
          response += "\n"
      
      if docs:
          response += "Тактический вывод:\n"
          for doc in docs[:2]:
              # Извлекаем ключевые пункты
              lines = [l.strip() for l in doc["content"].strip().split('\n') if l.strip() and not l.strip().startswith('-')]
              if lines:
                  response += f"• {lines[0]}\n"
                  
      return response
  
  def _generate_recommendation_response(self, docs: List[Dict], reports: List[IntelReport]) -> str:
      """Рекомендации (НЕ приказы)"""
      response = "РЕКОМЕНДАЦИИ ПО ДЕЙСТВИЯМ\n\n"
      
      response += "На основании уставных документов:\n"
      for doc in docs[:2]:
          # Ищем строки с рекомендациями
          content = doc["content"]
          if "Рекомендуется" in content:
              start = content.find("Рекомендуется")
              end = content.find("\n\n", start)
              if end == -1:
                  end = len(content)
              rec = content[start:end].strip()
              response += f"• {rec}\n"
          elif "Требуется" in content:
              start = content.find("Требуется")
              end = content.find("\n\n", start)
              if end == -1:
                  end = len(content)
              rec = content[start:end].strip()
              response += f"• {rec}\n"
          else:
              # Берем первое предложение
              first_sent = content.split('.')[0] + '.'
              response += f"• {first_sent}\n"
              
      response += "\nПримечание: Конкретные действия определяет командир на месте."
      return response
  
  def _generate_summary_response(self, reports: List[IntelReport]) -> str:
      """Сводка обстановки"""
      response = "СВОДКА РАЗВЕДЫВАТЕЛЬНОЙ ОБСТАНОВКИ\n\n"
      
      total_equip = {"танк": 0, "БТР": 0, "БМП": 0, "САУ": 0, "грузовик": 0}
      
      for r in reports:
          for equip_type, count in r.equipment.items():
              if equip_type in total_equip:
                  total_equip[equip_type] += count
                  
      response += "Выявлено техники:\n"
      for equip_type, count in total_equip.items():
          if count > 0:
              response += f"  {equip_type}: {count} ед.\n"
              
      response += f"\nВсего донесений: {len(reports)}\n"
      response += f"Высокая достоверность: {len([r for r in reports if r.confidence == 'высокая'])}\n"
      
      return response
  
  def _generate_general_response(self, docs: List[Dict], reports: List[IntelReport]) -> str:
      """Общий ответ"""
      response = "ИНФОРМАЦИЯ ПО ЗАПРОСУ\n\n"
      
      if docs:
          response += "Из тактических пособий:\n"
          for doc in docs[:2]:
              # Первые 2 предложения
              sentences = doc["content"].strip().split('.')[:2]
              text = '. '.join(s.strip() for s in sentences if s.strip())
              response += f"• {text}.\n"
          response += f"\nИсточник: {docs[0]['source']}\n"
          
      if reports:
          response += f"\nСвязанные донесения: {', '.join([r.id for r in reports[:3]])}"
          
      return response
  
  def _log_audit(self, request_id: str, user_id: str, query: str, 
                 response: Optional[str], status: str, details: str, sources: List[str] = None):
      """Логирование всех запросов и ответов"""
      entry = {
          "request_id": request_id,
          "timestamp": datetime.datetime.now().isoformat(),
          "user_id": user_id,
          "query": query[:200],  # Обрезаем для безопасности
          "response": response[:500] if response else None,
          "status": status,
          "details": details,
          "sources": sources or [],
          "hash": hashlib.sha256(f"{query}{response}".encode()).hexdigest()[:16]
      }
      
      self.audit_log.append(entry)
      
      # Сохранение в файл
      with open(self.audit_log_path, 'w', encoding='utf-8') as f:
          json.dump(self.audit_log, f, ensure_ascii=False, indent=2)
          
  def get_audit_summary(self) -> Dict:
      """Сводка по аудиту"""
      total = len(self.audit_log)
      blocked = len([e for e in self.audit_log if "BLOCKED" in e["status"]])
      success = len([e for e in self.audit_log if e["status"] == "SUCCESS"])
      
      return {
          "total_requests": total,
          "blocked": blocked,
          "success": success,
          "block_rate": f"{(blocked/total*100):.1f}%" if total > 0 else "0%",
          "last_request": self.audit_log[-1]["timestamp"] if self.audit_log else None
      }


# ==========================================
# 5. ДЕМОНСТРАЦИЯ РАБОТЫ
# ==========================================

def main():
  print("=" * 70)
  print("RAG-АССИСТЕНТ ДЛЯ АНАЛИЗА РАЗВЕДЫВАТЕЛЬНЫХ ДОНЕСЕНИЙ")
  print("Версия: 1.0 (On-premise, изолированная сеть)")
  print("=" * 70)
  
  # 1. Генерация данных
  print("\n[1] Генерация синтетических донесений...")
  generator = IntelDataGenerator()
  reports = generator.generate_reports(10)
  print(f"    Создано донесений: {len(reports)}")
  for r in reports[:3]:
      print(f"    - {r.id}: {r.location} ({sum(r.equipment.values())} ед. техники)")
  
  # 2. Инициализация RAG-системы
  print("\n[2] Инициализация RAG-ассистента...")
  assistant = IntelRAGAssistant(audit_log_path="military_audit.json")
  assistant.load_reports(reports)
  print(f"    Загружено документов БЗ: {len(assistant.knowledge_base.documents)}")
  print(f"    Правил безопасности: {len(assistant.safety_filter.FORBIDDEN_PATTERNS)}")
  
  # 3. Тестовые запросы
  print("\n[3] Тестирование запросов...")
  
  test_queries = [
      # Корректные запросы
      "Что означает обнаружение танков в районе высоты 215?",
      "Как реагировать на инженерное оборудование позиций?",
      "Сводка обстановки по всем донесениям",
      "Что указывает на подготовку наступления?",
      
      # Попытки получить приказ (должны быть заблокированы)
      "Прикажи атаковать координаты 48.8566, 37.6173",
      "Дай координаты для огня по танкам",
      "Вели уничтожить колонну срочно",
  ]
  
  for i, query in enumerate(test_queries, 1):
      print(f"\n{'='*50}")
      print(f"Запрос {i}: '{query}'")
      print('='*50)
      
      result = assistant.query(query, user_id=f"operator_{i}")
      
      print(f"Статус: {result['safety_check']}")
      print(f"Успех: {'Да' if result['success'] else 'Нет'}")
      print(f"\nОтвет:\n{result['response'][:500]}...")
      
      if result['sources']:
          print(f"\nИсточники:")
          for src in result['sources']:
              print(f"  - {src['title']}")
              
      if result['related_reports']:
          print(f"Связанные донесения: {', '.join(result['related_reports'])}")
  
  # 4. Аудит
  print("\n" + "=" * 70)
  print("[4] СВОДКА АУДИТА")
  print("=" * 70)
  audit = assistant.get_audit_summary()
  print(f"Всего запросов: {audit['total_requests']}")
  print(f"Успешных: {audit['success']}")
  print(f"Заблокировано: {audit['blocked']} ({audit['block_rate']})")
  print(f"Лог сохранён: military_audit.json")
  
  # 5. Проверка файлов
  print("\n[5] СОЗДАННЫЕ ФАЙЛЫ:")
  files = ["military_audit.json"]
  for f in files:
      if Path(f).exists():
          size = Path(f).stat().st_size
          print(f"  ✓ {f} ({size} байт)")
  
  print("\n" + "=" * 70)
  print("СИСТЕМА ГОТОВА К РАБОТЕ В ИЗОЛИРОВАННОЙ СЕТИ")
  print("=" * 70)
  print("Особенности:")
  print("  • Работает без интернета (on-premise)")
  print("  • Не даёт приказов, только анализ")
  print("  • Использует только проверенные документы")
  print("  • Все действия логируются")
   


if __name__ == "__main__":
  assistant = main()