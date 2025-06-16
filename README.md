import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Физиологические константы
Cm = 1.0      # мкФ/см^2
R = 8314.0    # [Дж/(кмоль·K)]
F = 96485.0   # Постоянная Фарадея [Кл/кмоль]
T = 310.0     # Температура [K]

# Внеклеточные концентрации [ммоль/л]
Na_o = 140.0
K_o = 5.4
Ca_o = 1.8

# Начальные внутриклеточные концентрации [ммоль/л]
Na_i = 18.0
K_i = 145.0
Ca_i = 0.0002

gNa = 23.0    # мСм/см^2
gK = 0.282 * np.sqrt(K_o / 5.4)   # мСм/см^2
gK1 = 0.6047 * np.sqrt(K_o / 5.4)  # мСм/см^2
gKp = 0.0183  # мСм/см^2
gCaL = 0.009  # мСм/см^2

ENa = (R * T / F) * np.log(Na_o / Na_i)
EK = (R * T / F) * np.log(K_o / K_i)
ECa = (R * T / (2 * F)) * np.log(Ca_o / Ca_i)

# Уравнения воротных переменных
def alpha_m(V): return 0.32 * (V + 47.13) / (1 - np.exp(-0.1 * (V + 47.13)))
def beta_m(V): return 0.08 * np.exp(-V / 11)
def alpha_h(V): return 0.135 * np.exp(-(V + 80) / 6.8)
def beta_h(V): return 3.56 * np.exp(0.079 * V) + 310000 * np.exp(0.35 * V)
def alpha_j(V): return ((-127140 * np.exp(0.2444 * V) - 3.474e-5 * np.exp(-0.04391 * V)) * (V + 37.78) / (1 + np.exp(0.311 * (V + 79.23))))
def beta_j(V): return 0.1212 * np.exp(-0.01052 * V) / (1 + np.exp(-0.1378 * (V + 40.14)))

def alpha_d(V): return 0.095 * np.exp(-0.01 * (V - 5)) / (1 + np.exp(-0.072 * (V - 5)))
def beta_d(V): return 0.07 * np.exp(-0.017 * (V + 44)) / (1 + np.exp(0.05 * (V + 44)))
def alpha_f(V): return 0.012 * np.exp(-0.008 * (V + 28)) / (1 + np.exp(0.15 * (V + 28)))
def beta_f(V): return 0.0065 * np.exp(-0.02 * (V + 30)) / (1 + np.exp(-0.2 * (V + 30)))

def alpha_X(V): return 0.0005 * np.exp(0.083 * (V + 50)) / (1 + np.exp(0.057 * (V + 50)))
def beta_X(V): return 0.0013 * np.exp(-0.06 * (V + 20)) / (1 + np.exp(-0.04 * (V + 20)))

# Внешний ток стимуляции
def I_stim(t):
    if 0 < t < 1.5:
        return 80
    return 0.0

# Основная система ОДУ
def lr91_rhs(t, y):
    V, m, h, j, d, f, X = y

    am = alpha_m(V)
    bm = beta_m(V)
    ah = alpha_h(V)
    bh = beta_h(V)
    aj = alpha_j(V)
    bj = beta_j(V)
    ad = alpha_d(V)
    bd = beta_d(V)
    af = alpha_f(V)
    bf = beta_f(V)
    aX = alpha_X(V)
    bX = beta_X(V)

    dm = am * (1 - m) - bm * m
    dh = ah * (1 - h) - bh * h
    dj = aj * (1 - j) - bj * j
    dd = ad * (1 - d) - bd * d
    df = af * (1 - f) - bf * f
    dX = aX * (1 - X) - bX * X

    INa = gNa * (m**3) * h * j * (V - ENa)
    ICaL = gCaL * d * f * (V - ECa)
    IK = gK * X**2 * (V - EK)
    IK1 = gK1 * (V - EK) / (1 + np.exp(1.31 * (V - EK - 12)))
    Kp = 1 / (1 + np.exp((7.488 - V) / 5.98))
    IKp = gKp * Kp * (V - EK)

    Iion = INa + ICaL + IK + IK1 + IKp
    dV = -(Iion - I_stim(t)) / Cm

    return [dV, dm, dh, dj, dd, df, dX]

# Начальные условия
V0 = -84.0
y0 = [V0, 0.0, 0.75, 0.75, 0.0, 1.0, 0.0]

# Время симуляции
t_span = (0, 400)
t_eval = np.linspace(*t_span, 4000)

# Решение
sol = solve_ivp(lr91_rhs, t_span, y0, method='BDF', t_eval=t_eval)

# Получение данных
V = sol.y[0]
t = sol.t

# Нахождение пиков (потенциалов действия)
peaks, _ = find_peaks(V, height=0)
if len(peaks) == 0:
    raise ValueError("Потенциалы действия не найдены. Проверьте параметры модели.")

# Выбор первого пика для анализа
peak_idx = peaks[0]
peak_time = t[peak_idx]
peak_voltage = V[peak_idx]

# 1. Находим точку начала реполяризации (когда dV/dt становится отрицательным после пика)
dvdt = np.diff(V) / np.diff(t)
repolarization_start = peak_idx + np.where(dvdt[peak_idx:] < 0)[0][0]

# 2. Уровень 90% реполяризации: 90% от амплитуды (пик - потенциал покоя)
resting_potential = V0
amplitude = peak_voltage - resting_potential
repolarization_level = peak_voltage - 0.9 * amplitude

# 3. Находим момент пересечения этого уровня после пика
crossing_points = np.where(V[repolarization_start:] <= repolarization_level)[0]
if len(crossing_points) == 0:
    apd90 = np.nan
else:
    apd90 = t[repolarization_start + crossing_points[0]] - t[repolarization_start]


# Вычисление Vmax (максимальная скорость деполяризации)
dV = np.diff(V) / np.diff(t)
vmax = np.max(dV)

# Максимальное значение потенциала
max_voltage = np.max(V)

# Построение графика
plt.figure(figsize=(12, 6))
plt.plot(t, V)
plt.title("Потенциал действия при изменении амплитуды стимула (20) (Luo-Rudy 1991)")
plt.xlabel("Время (мс)")
plt.ylabel("Мембранный потенциал (мВ)")
plt.grid()
plt.show()

# Вывод параметров в консоль
print(f"Максимальное значение потенциала: {max_voltage:.1f} мВ")
print(f"APD90: {apd90:.1f} мс")
print(f"Vmax: {vmax:.1f} В/с")
