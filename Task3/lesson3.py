import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import os

# ============================================
# БИТВА ЗА ВЫСОТУ 217.3
# Тактическое моделирование методом матричных игр
# ============================================

print("=" * 70)
print("БИТВА ЗА ВЫСОТУ 217.3 - ТАКТИЧЕСКОЕ МОДЕЛИРОВАНИЕ")
print("=" * 70)

# ---------------------------------------------------------
# ИСХОДНЫЕ ДАННЫЕ БОЕВЫХ ПОДРАЗДЕЛЕНИЙ
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ИСХОДНЫЕ ДАННЫЕ")
print("=" * 70)

forces_A = {
    "name": "Сторона А (мотострелки)",
    "units": [
        {"type": "Взвод мотострелков", "count": 1, "firepower": 60, "armor": 30, "mobility": 40},
        {"type": "БТР", "count": 1, "firepower": 40, "armor": 50, "mobility": 70}
    ],
    "total_firepower": 100,
    "total_armor": 80,
    "total_mobility": 110
}

forces_B = {
    "name": "Сторона Б (спецназ)",
    "units": [
        {"type": "Отделение спецназа", "count": 1, "firepower": 45, "stealth": 80, "mobility": 60},
        {"type": "Гранатометный расчет", "count": 1, "firepower": 70, "stealth": 20, "mobility": 30}
    ],
    "total_firepower": 115,
    "total_stealth": 100,
    "total_mobility": 90
}

print(f"\n{forces_A['name']}:")
for unit in forces_A["units"]:
    print(f"  • {unit['type']}: огневая мощь={unit['firepower']}, броня={unit['armor']}, мобильность={unit['mobility']}")

print(f"\n{forces_B['name']}:")
for unit in forces_B["units"]:
    stealth = unit.get('stealth', 'N/A')
    print(f"  • {unit['type']}: огневая мощь={unit['firepower']}, маскировка={stealth}, мобильность={unit['mobility']}")

# ---------------------------------------------------------
# СТРАТЕГИИ СТОРОН
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ВАРИАНТЫ ДЕЙСТВИЙ (СТРАТЕГИИ)")
print("=" * 70)

strategies_A = [
    "Атака фронтально",
    "Обход справа", 
    "Обход слева",
    "Артиллерийский обстрел"
]

strategies_B = [
    "Оборона фронтальная",
    "Оборона правого фланга",
    "Оборона левого фланга", 
    "Контратака",
    "Засада в низине"
]

print(f"\nСторона А (наступление):")
for i, s in enumerate(strategies_A, 1):
    print(f"  {i}. {s}")

print(f"\nСторона Б (оборона):")
for i, s in enumerate(strategies_B, 1):
    print(f"  {i}. {s}")

# ---------------------------------------------------------
# РАСЧЁТ ПЛАТЁЖНОЙ МАТРИЦЫ (боевые потери и эффективность)
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("РАСЧЁТ ТАКТИЧЕСКОЙ ЭФФЕКТИВНОСТИ")
print("=" * 70)

def calculate_effectiveness(strat_A, strat_B, forces_A, forces_B):
    """
    Расчёт эффективности действий на основе тактических факторов
    Возвращает (потери_Б, потери_А) - чем меньше потери своих, тем лучше
    """
    
    # Базовые характеристики
    fire_A = forces_A["total_firepower"]
    armor_A = forces_A["total_armor"]
    mob_A = forces_A["total_mobility"]
    
    fire_B = forces_B["total_firepower"]
    stealth_B = forces_B["total_stealth"]
    mob_B = forces_B["total_mobility"]
    
    # Коэффициенты в зависимости от стратегий
    
    # Сторона А
    if strat_A == "Атака фронтально":
        coef_fire_A = 1.0
        coef_armor_A = 1.0
        coef_mob_A = 0.5
        surprise_A = 0
    elif strat_A == "Обход справа":
        coef_fire_A = 0.8
        coef_armor_A = 0.7
        coef_mob_A = 1.2
        surprise_A = 0.3
    elif strat_A == "Обход слева":
        coef_fire_A = 0.8
        coef_armor_A = 0.7
        coef_mob_A = 1.2
        surprise_A = 0.3
    else:  # Артиллерийский обстрел
        coef_fire_A = 1.5
        coef_armor_A = 0
        coef_mob_A = 0
        surprise_A = 0.5
    
    # Сторона Б
    if strat_B == "Оборона фронтальная":
        coef_fire_B = 1.2
        coef_defense_B = 1.3
        ambush_B = 0
    elif strat_B == "Оборона правого фланга":
        coef_fire_B = 1.0
        coef_defense_B = 1.1
        ambush_B = 0.2 if "справа" in strat_A else 0
    elif strat_B == "Оборона левого фланга":
        coef_fire_B = 1.0
        coef_defense_B = 1.1
        ambush_B = 0.2 if "слева" in strat_A else 0
    elif strat_B == "Контратака":
        coef_fire_B = 1.1
        coef_defense_B = 0.8
        ambush_B = 0
    else:  # Засада в низине
        coef_fire_B = 1.3
        coef_defense_B = 0.9
        ambush_B = 0.4 if strat_A != "Артиллерийский обстрел" else 0
    
    # Расчёт урона
    # Урон А по Б
    if strat_A == "Артиллерийский обстрел":
        damage_to_B = fire_A * coef_fire_A * (1 - stealth_B/200)  # Маскировка снижает урон
        damage_to_A = 0  # Нет прямого контакта
    else:
        damage_to_B = fire_A * coef_fire_A * (1 + surprise_A) / coef_defense_B
        damage_to_A = fire_B * coef_fire_B * (1 + ambush_B) / (armor_A/100 * coef_armor_A)
    
    # Нормализация к шкале 0-100 (потери в процентах)
    losses_B = min(100, damage_to_B / 2)
    losses_A = min(100, damage_to_A / 2)
    
    # Эффективность = разница потерь (потери противника - свои потери)
    # Положительное значение - выигрыш А, отрицательное - выигрыш Б
    effectiveness = losses_B - losses_A
    
    return effectiveness, losses_A, losses_B

# Построение матрицы
n_A = len(strategies_A)
n_B = len(strategies_B)

payoff_matrix = np.zeros((n_A, n_B))
losses_A_matrix = np.zeros((n_A, n_B))
losses_B_matrix = np.zeros((n_A, n_B))

for i, strat_A in enumerate(strategies_A):
    for j, strat_B in enumerate(strategies_B):
        eff, loss_A, loss_B = calculate_effectiveness(strat_A, strat_B, forces_A, forces_B)
        payoff_matrix[i, j] = eff
        losses_A_matrix[i, j] = loss_A
        losses_B_matrix[i, j] = loss_B

print("\nПлатёжная матрица (эффективность А - разница потерь):")
print("-" * 80)
print(f"{'А \\ Б':<25}", end="")
for s in strategies_B:
    print(f"{s[:12]:<12}", end="")
print()
print("-" * 80)

for i, strat_A in enumerate(strategies_A):
    print(f"{strat_A:<25}", end="")
    for j in range(n_B):
        print(f"{payoff_matrix[i,j]:>+10.1f}", end="  ")
    print()

# ---------------------------------------------------------
# ПОИСК РАВНОВЕСИЯ НЭША
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("АНАЛИЗ РАВНОВЕСИЯ НЭША")
print("=" * 70)

def find_nash_equilibrium(payoff_matrix):
    """Поиск равновесия Нэша в чистых стратегиях"""
    n_A, n_B = payoff_matrix.shape
    nash_points = []
    
    for i in range(n_A):
        for j in range(n_B):
            # Проверяем, является ли (i,j) равновесием
            # Сторона А не хочет отклоняться
            best_for_A = True
            for alt_i in range(n_A):
                if payoff_matrix[alt_i, j] > payoff_matrix[i, j]:
                    best_for_A = False
                    break
            
            # Сторона Б не хочет отклоняться (минимизирует выигрыш А)
            best_for_B = True
            for alt_j in range(n_B):
                if payoff_matrix[i, alt_j] < payoff_matrix[i, j]:
                    best_for_B = False
                    break
            
            if best_for_A and best_for_B:
                nash_points.append((i, j, payoff_matrix[i, j]))
    
    return nash_points

nash_points = find_nash_equilibrium(payoff_matrix)

print("\nРавновесия Нэша в чистых стратегиях:")
if nash_points:
    for i, j, val in nash_points:
        print(f"  {strategies_A[i]} vs {strategies_B[j]} → эффективность {val:+.1f}")
else:
    print("  Нет равновесий в чистых стратегиях")

# Минимакс для стороны А (максимин)
maximin_A = np.max(np.min(payoff_matrix, axis=1))
maximin_idx_A = np.argmax(np.min(payoff_matrix, axis=1))
print(f"\nМаксимин для А: {maximin_A:+.1f} (стратегия '{strategies_A[maximin_idx_A]}')")

# Минимакс для стороны Б (минимакс)
minimax_B = np.min(np.max(payoff_matrix, axis=0))
minimax_idx_B = np.argmin(np.max(payoff_matrix, axis=0))
print(f"Минимакс для Б: {minimax_B:+.1f} (стратегия '{strategies_B[minimax_idx_B]}')")

if maximin_A == minimax_B:
    print(f"\nЦена игры: {maximin_A:+.1f}")
else:
    print(f"\nИнтервал цены игры: [{maximin_A:+.1f}, {minimax_B:+.1f}]")

# ---------------------------------------------------------
# СМЕШАННЫЕ СТРАТЕГИИ (линейное программирование)
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("РАСЧЁТ СМЕШАННЫХ СТРАТЕГИЙ")
print("=" * 70)

def solve_mixed_strategy(payoff_matrix, player='A'):
    """
    Решение для смешанных стратегий методом линейного программирования
    """
    from scipy.optimize import linprog
    
    m, n = payoff_matrix.shape
    
    if player == 'A':
        # Максимизируем v (минимальный выигрыш)
        # min -v
        # s.t. sum(p_i * a_ij) >= v для всех j
        # sum(p_i) = 1, p_i >= 0
        
        c = [0] * m + [-1]  # Коэффициенты: p_1...p_m, -v
        
        A_ub = []
        b_ub = []
        for j in range(n):
            row = list(-payoff_matrix[:, j]) + [1]  # -a_ij * p_i + v <= 0
            A_ub.append(row)
            b_ub.append(0)
        
        A_eq = [[1] * m + [0]]  # sum(p_i) = 1
        b_eq = [1]
        
        bounds = [(0, 1) for _ in range(m)] + [(None, None)]  # p_i в [0,1], v неограничена
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            probs = result.x[:-1]
            value = -result.x[-1]
            return probs, value
        else:
            return None, None
    else:
        # Для игрока Б (минимизация)
        # Аналогично, но транспонируем матрицу и меняем знаки
        return solve_mixed_strategy(-payoff_matrix.T, 'A')

try:
    probs_A, value_A = solve_mixed_strategy(payoff_matrix, 'A')
    if probs_A is not None:
        print(f"\nОптимальная смешанная стратегия А:")
        for i, p in enumerate(probs_A):
            if p > 0.001:
                print(f"  {strategies_A[i]}: {p*100:.1f}%")
        print(f"  Цена игры: {value_A:+.1f}")
    
    probs_B, value_B = solve_mixed_strategy(payoff_matrix, 'B')
    if probs_B is not None:
        print(f"\nОптимальная смешанная стратегия Б:")
        for i, p in enumerate(probs_B):
            if p > 0.001:
                print(f"  {strategies_B[i]}: {p*100:.1f}%")
        print(f"  Цена игры: {value_B:+.1f}")
        
except Exception as e:
    print(f"\nОшибка расчёта смешанных стратегий: {e}")
    print("Возможно, требуется установка scipy: pip install scipy")

# ---------------------------------------------------------
# АНАЛИЗ ПОТЕРЬ
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("АНАЛИЗ БОЕВЫХ ПОТЕРЬ")
print("=" * 70)

print("\nПотери стороны А (%):")
print("-" * 80)
print(f"{'А \\ Б':<25}", end="")
for s in strategies_B:
    print(f"{s[:10]:<10}", end="")
print()
print("-" * 80)

for i, strat_A in enumerate(strategies_A):
    print(f"{strat_A:<25}", end="")
    for j in range(n_B):
        print(f"{losses_A_matrix[i,j]:>8.1f}", end="  ")
    print()

print("\nПотери стороны Б (%):")
print("-" * 80)
print(f"{'А \\ Б':<25}", end="")
for s in strategies_B:
    print(f"{s[:10]:<10}", end="")
print()
print("-" * 80)

for i, strat_A in enumerate(strategies_A):
    print(f"{strat_A:<25}", end="")
    for j in range(n_B):
        print(f"{losses_B_matrix[i,j]:>8.1f}", end="  ")
    print()

# ---------------------------------------------------------
# ВИЗУАЛИЗАЦИЯ
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ВИЗУАЛИЗАЦИЯ ТАКТИЧЕСКОЙ СИТУАЦИИ")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# Создаём сетку для сложной компоновки
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# --- График 1: Платёжная матрица (тепловая карта) ---
ax1 = fig.add_subplot(gs[0, :2])
im1 = ax1.imshow(payoff_matrix, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=50)
ax1.set_xticks(range(n_B))
ax1.set_yticks(range(n_A))
ax1.set_xticklabels([s[:12] for s in strategies_B], rotation=45, ha='right')
ax1.set_yticklabels([s[:15] for s in strategies_A])
ax1.set_title('Платёжная матрица (эффективность А)', fontsize=12, fontweight='bold')

# Добавляем значения
for i in range(n_A):
    for j in range(n_B):
        text = ax1.text(j, i, f'{payoff_matrix[i, j]:+.0f}',
                       ha="center", va="center", 
                       color="white" if abs(payoff_matrix[i, j]) > 25 else "black",
                       fontweight='bold', fontsize=9)
plt.colorbar(im1, ax=ax1, label='Эффективность А')

# --- График 2: Потери обеих сторон ---
ax2 = fig.add_subplot(gs[0, 2])
x = np.arange(n_A)
width = 0.35

# Усреднённые потери по всем стратегиям противника
avg_losses_A = np.mean(losses_A_matrix, axis=1)
avg_losses_B = np.mean(losses_B_matrix, axis=1)

bars1 = ax2.barh(x - width/2, avg_losses_A, width, label='Потери А', color='blue', alpha=0.7)
bars2 = ax2.barh(x + width/2, avg_losses_B, width, label='Потери Б', color='red', alpha=0.7)

ax2.set_yticks(x)
ax2.set_yticklabels([s[:15] for s in strategies_A])
ax2.set_xlabel('Средние потери (%)')
ax2.set_title('Сравнение потерь', fontsize=12, fontweight='bold')
ax2.legend()

# --- График 3: Тактическая карта высоты ---
ax3 = fig.add_subplot(gs[1, :])
ax3.set_xlim(0, 100)
ax3.set_ylim(0, 100)
ax3.set_aspect('equal')
ax3.axis('off')
ax3.set_title('Тактическая схема высоты 217.3', fontsize=14, fontweight='bold')

# Рисуем высоту (треугольник)
from matplotlib.patches import Polygon
height = Polygon([(30, 20), (70, 20), (50, 80)], 
                 facecolor='sandybrown', edgecolor='saddlebrown', linewidth=3)
ax3.add_patch(height)
ax3.text(50, 50, 'ВЫСОТА\n217.3', ha='center', va='center', 
         fontsize=14, fontweight='bold', color='saddlebrown')

# Позиции сторон
# Сторона А (юг)
ax3.add_patch(plt.Rectangle((20, 5), 60, 10, facecolor='lightblue', 
                            edgecolor='blue', linewidth=2, alpha=0.7))
ax3.text(50, 10, 'СТОРОНА А (мотострелки + БТР)', ha='center', va='center',
         fontsize=10, fontweight='bold', color='darkblue')

# Сторона Б (север)
ax3.add_patch(plt.Rectangle((35, 85), 30, 10, facecolor='lightcoral', 
                            edgecolor='darkred', linewidth=2, alpha=0.7))
ax3.text(50, 90, 'СТОРОНА Б (спецназ)', ha='center', va='center',
         fontsize=10, fontweight='bold', color='darkred')

# Направления атаки (стрелки)
arrows = [
    (50, 15, 50, 25, 'Фронтально', 'blue'),
    (70, 15, 65, 25, 'Справа', 'green'),
    (30, 15, 35, 25, 'Слева', 'green'),
    (10, 50, 25, 50, 'Арт. огонь', 'purple')
]

for x1, y1, x2, y2, label, color in arrows:
    ax3.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    ax3.text(x1, y1-3, label, ha='center', fontsize=8, color=color)

# Легенда
ax3.text(5, 95, 'Легенда:', fontsize=10, fontweight='bold')
ax3.text(5, 92, '→ Фронтальная атака', fontsize=8, color='blue')
ax3.text(5, 89, '→ Обходные манёвры', fontsize=8, color='green')
ax3.text(5, 86, '→ Артиллерия', fontsize=8, color='purple')

# --- График 4: Доминирование стратегий ---
ax4 = fig.add_subplot(gs[2, 0])

# Проверка доминирования для стороны А
dominance_A = np.zeros((n_A, n_A))
for i in range(n_A):
    for k in range(n_A):
        if i != k and all(payoff_matrix[i, :] >= payoff_matrix[k, :]) and any(payoff_matrix[i, :] > payoff_matrix[k, :]):
            dominance_A[i, k] = 1  # i доминирует k

im4 = ax4.imshow(dominance_A, cmap='Blues', aspect='auto')
ax4.set_xticks(range(n_A))
ax4.set_yticks(range(n_A))
ax4.set_xticklabels([f'A{i+1}' for i in range(n_A)])
ax4.set_yticklabels([f'A{i+1}' for i in range(n_A)])
ax4.set_title('Доминирование стратегий А', fontsize=11, fontweight='bold')
ax4.set_xlabel('Доминируемая')
ax4.set_ylabel('Доминирующая')

for i in range(n_A):
    for j in range(n_A):
        if dominance_A[i, j] == 1:
            ax4.text(j, i, 'D', ha='center', va='center', color='white', fontweight='bold')

# --- График 5: Риски и неопределённость ---
ax5 = fig.add_subplot(gs[2, 1])

# Диапазон возможных исходов для каждой стратегии А
for i, strat in enumerate(strategies_A):
    min_val = np.min(payoff_matrix[i, :])
    max_val = np.max(payoff_matrix[i, :])
    mean_val = np.mean(payoff_matrix[i, :])
    
    ax5.plot([min_val, max_val], [i, i], 'b-', linewidth=4, alpha=0.5)
    ax5.scatter([mean_val], [i], color='red', s=50, zorder=5)
    ax5.scatter([min_val, max_val], [i, i], color='blue', s=30, zorder=5)

ax5.set_yticks(range(n_A))
ax5.set_yticklabels([s[:15] for s in strategies_A])
ax5.set_xlabel('Эффективность')
ax5.set_title('Диапазон исходов (min-max)', fontsize=11, fontweight='bold')
ax5.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax5.grid(True, alpha=0.3, axis='x')

# --- График 6: Рекомендации ---
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
ax6.set_title('ТАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ', fontsize=11, fontweight='bold')

recommendations = f"""
ДЛЯ СТОРОНЫ А (Наступление):

1. {"ИЗБЕГАТЬ" if np.mean(payoff_matrix[0, :]) < 0 else "ИСПОЛЬЗОВАТЬ"} фронтальной атаки
   Средняя эффективность: {np.mean(payoff_matrix[0, :]):+.1f}

2. {"ПРЕДПОЧТИТЕЛЬНЫ" if np.mean(payoff_matrix[1, :]) > 0 else "ОСТОРОЖНО"} обходы
   Справа: {np.mean(payoff_matrix[1, :]):+.1f}
   Слева: {np.mean(payoff_matrix[2, :]):+.1f}

3. Артиллерия эффективна против:
   - Засады ({payoff_matrix[3, 4]:+.1f})
   - Фронтальной обороны ({payoff_matrix[3, 0]:+.1f})

ДЛЯ СТОРОНЫ Б (Оборона):

1. Оптимальная стратегия: 
   {strategies_B[np.argmax(np.min(payoff_matrix, axis=0))]}

2. Избегать фронтальной обороны
   против артиллерии

3. Контратака рискована, но
   эффективна против обходов
"""

ax6.text(0.05, 0.95, recommendations, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()

# Сохранение
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'tactical_height_217_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nГрафик сохранён: {output_path}")
plt.show()

# ---------------------------------------------------------
# ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ (Монте-Карло)
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ (МЕТОД МОНТЕ-КАРЛО)")
print("=" * 70)

np.random.seed(42)
n_simulations = 10000

# Случайный выбор стратегий
random_strategies_A = np.random.choice(n_A, n_simulations)
random_strategies_B = np.random.choice(n_B, n_simulations)

results = []
for sa, sb in zip(random_strategies_A, random_strategies_B):
    # Добавляем случайный шум (неопределённость боя)
    noise = np.random.normal(0, 10)
    eff = payoff_matrix[sa, sb] + noise
    results.append({
        'strat_A': strategies_A[sa],
        'strat_B': strategies_B[sb],
        'effectiveness': eff,
        'winner': 'A' if eff > 0 else 'B' if eff < 0 else 'draw'
    })

# Статистика по стратегиям
print(f"\nРезультаты {n_simulations} симуляций:")
print(f"Победы А: {sum(1 for r in results if r['winner'] == 'A')} ({sum(1 for r in results if r['winner'] == 'A')/n_simulations*100:.1f}%)")
print(f"Победы Б: {sum(1 for r in results if r['winner'] == 'B')} ({sum(1 for r in results if r['winner'] == 'B')/n_simulations*100:.1f}%)")
print(f"Ничьи: {sum(1 for r in results if r['winner'] == 'draw')} ({sum(1 for r in results if r['winner'] == 'draw')/n_simulations*100:.1f}%)")

# ---------------------------------------------------------
# ВЫВОДЫ
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ВЫВОДЫ ДЛЯ КУРСАНТОВ")
print("=" * 70)

conclusions = """
1. ТАКТИЧЕСКАЯ ГИБКОСТЬ:
   Чистая стратегия редко оптимальна. Смешанные стратегии 
   (вероятностный выбор) затрудняют прогнозирование противником.

2. РОЛЬ РАЗВЕДКИ:
   Незнание стратегии противника увеличивает потери на 20-30%.
   Разведка должна определять: обходят ли нас, идут фронтально 
   или готовят артподготовку.

3. СИНЕРГИЯ ВОЙСК:
   Мотострелки + БТР: сила в мобильности и огневой мощи
   Спецназ + гранатомёты: сила в скрытности и точном ударе
   
   РазведкаБТР нейтрализует скрытность спецназа.
   Гранатомёты эффективны против бронетехники при засаде.

4. ДОМИНИРОВАНИЕ:
   Если одна стратегия доминирует другую - исключить худшую.
   Но на практике доминирование редко бывает строгим.

5. РИСК-МЕНЕДЖМЕНТ:
   Минимаксная стратегия гарантирует результат не хуже цены игры.
   Но может упустить возможность крупного выигрыша.

6. ПРАКТИЧЕСКОЕ ПРИМЕНЕНИЕ:
   - Для А: комбинировать артподготовку с обходом
   - Для Б: держать резерв для контратаки на флангах
   - Всегда иметь план отхода (стратегия не сработала)
"""

print(conclusions)

print("\n" + "=" * 70)
print("КОД ВЫПОЛНЕН УСПЕШНО")
print("=" * 70)