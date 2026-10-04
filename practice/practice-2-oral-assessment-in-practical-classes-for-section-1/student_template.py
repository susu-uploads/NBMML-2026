
# ============================================================================
# ПР 1 (повышенный уровень): Идентификация закона распределения
# с применением генеративной нейронной сети
#
# ШАБЛОН КОДА ДЛЯ САМОСТОЯТЕЛЬНОГО ВЫПОЛНЕНИЯ
#
# Инструкция:
# 1. Заполните все ячейки, помеченные комментарием  # <-- ЗАПОЛНИТЬ
# 2. Не используйте готовые функции подбора распределения
#    (scipy.stats.fit, scipy.stats.rv_continuous.fit) для основных расчётов.
#    Они разрешены ТОЛЬКО в блоке верификации (раздел 6) для самопроверки.
# 3. Все числовые характеристики, теоретические частоты и критерии
#    рассчитывайте вручную по формулам.
# 4. Результаты сохраняйте в таблицы и файлы.
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

# Зафиксируйте random seed для воспроизводимости
SEED = 42  # <-- ЗАПОЛНИТЬ: можете изменить
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# РАЗДЕЛ 1. ИСХОДНЫЕ ДАННЫЕ И ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ
# ============================================================================

# --- 1.1. Исходная выборка (50 реальных значений) ---
# Вариант: рост студентов, см (из базовой ПР 1)
# Если у вас другой вариант — замените значения на свои

data_raw = np.array([
    173, 180, 175, 178, 170, 181, 179, 175, 183, 182,
    170, 177, 181, 175, 178, 181, 179, 178, 179, 185,
    180, 178, 181, 180, 171, 182, 174, 180, 179, 180,
    180, 172, 176, 181, 179, 180, 179, 178, 184, 185,
    189, 187, 184, 186, 179, 188, 192, 179, 186, 190
], dtype=np.float32)

n_original = len(data_raw)
print(f"Объём исходной выборки: {n_original}")

# --- 1.2. Предварительный разведочный анализ ---
# Рассчитайте вручную: среднее, дисперсию, СКО, коэффициент вариации,
# медиану, минимум, максимум, размах

# <-- ЗАПОЛНИТЬ: рассчитайте характеристики по формулам
mean_orig = ...        # X̄ = (1/n) * Σ x_i
var_orig = ...         # D = (1/(n-1)) * Σ (x_i - X̄)²
std_orig = ...         # σ = √D
cv_orig = ...          # V = (σ / X̄) * 100%
median_orig = ...
min_orig = ...
max_orig = ...
range_orig = ...      # R = x_max - x_min

print(f"Среднее: {mean_orig}")
print(f"Дисперсия: {var_orig}")
print(f"СКО: {std_orig}")
print(f"Коэффициент вариации: {cv_orig}%")
print(f"Медиана: {median_orig}")
print(f"Размах: {range_orig}")

# --- 1.3. Гистограмма исходных данных ---
fig, ax = plt.subplots(figsize=(8, 5))
# <-- ЗАПОЛНИТЬ: постройте гистограмму
# Подсказка: используйте ax.hist() с подходящим числом интервалов
# Формула Стерджеса: K = 1 + 3.3 * lg(n)

k_bins = ...  # <-- ЗАПОЛНИТЬ: число интервалов
ax.hist(data_raw, bins=k_bins, edgecolor='black', alpha=0.7)
ax.set_title("Гистограмма исходных данных")
ax.set_xlabel("Значения")
ax.set_ylabel("Частота")
plt.tight_layout()
plt.savefig("hist_original.png", dpi=150)
plt.close()
print("Гистограмма исходных данных сохранена в hist_original.png")


# ============================================================================
# РАЗДЕЛ 2. ОБУЧЕНИЕ VAE И ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================================

# --- 2.1. Нормализация данных в [0, 1] ---
# VAE работает лучше с нормализованными данными
data_min = data_raw.min()
data_max = data_raw.max()
data_norm = (data_raw - data_min) / (data_max - data_min)

# --- 2.2. Класс Variational Autoencoder (VAE) для одномерных данных ---

class VAE1D(nn.Module):
    """
    Вариационный автоэнкодер для одномерных данных.

    Архитектура:
      Энкодер: input(1) -> Linear(16) -> Linear(8) -> [mu(2), logvar(2)]
      Декодер: latent(2) -> Linear(8) -> Linear(16) -> output(1)

    Принцип работы:
      1. Энкодер получает значение x и сжимает его в латентное пространство,
         описываемое двумя числами: mu (среднее) и logvar (логарифм дисперсии).
      2. Из этого распределения сэмплируется точка z с помощью
         "трюка репараметризации": z = mu + sigma * epsilon,
         где epsilon ~ N(0, 1).
      3. Декодер восстанавливает значение из z.
    """

    def __init__(self):
        super(VAE1D, self).__init__()

        # Энкодер: сжимает вход (1 число) в латентное пространство (2 числа)
        self.encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, 2)       # среднее латентного распределения
        self.fc_logvar = nn.Linear(8, 2)   # логарифм дисперсии

        # Декодер: восстанавливает значение из латентного пространства
        self.decoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # выход в [0, 1], т.к. данные нормализованы
        )

    def encode(self, x):
        """Кодирование входа в латентное распределение (mu, logvar)."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        Трюк репараметризации: позволяет градиентам проходить через
        случайную выборку. Без него сеть нельзя обучить методом обратного
        распространения ошибки.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Декодирование из латентного пространства в значение данных."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# --- 2.3. Функция потерь VAE ---
# Состоит из двух частей:
#   1. MSE — насколько хорошо восстановлены данные (реконструкция)
#   2. KL-дивергенция — насколько латентное распределение близко к N(0, 1)

def vae_loss(recon_x, x, mu, logvar):
    """
    recon_x: восстановленные значения
    x:       исходные значения
    mu:      среднее латентного распределения
    logvar:  логарифм дисперсии латентного распределения
    """
    # MSE — среднеквадратичная ошибка реконструкции
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL-дивергенция — мера отличия латентного распределения от N(0, 1)
    # Формула: -0.5 * Σ (1 + logvar - mu² - exp(logvar))
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KL


# --- 2.4. Обучение VAE ---

# Подготовка данных: преобразуем в тензор PyTorch
train_tensor = torch.FloatTensor(data_norm).unsqueeze(1)  # shape: (50, 1)
train_dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True
)

# Создание модели и оптимизатора
vae = VAE1D()
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Цикл обучения
EPOCHS = 2000  # <-- ЗАПОЛНИТЬ: можно увеличить для лучшей сходимости
print(f"\nНачало обучения VAE ({EPOCHS} эпох)...")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch_x, _ in train_loader:
        optimizer.zero_grad()
        recon_x, mu, logvar = vae(batch_x)
        loss = vae_loss(recon_x, batch_x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 500 == 0:
        print(f"  Эпоха {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

print("Обучение завершено.")


# --- 2.5. Генерация новых данных ---

def generate_vae(model, n_samples, data_min, data_max):
    """
    Генерация новых данных из обученного VAE.

    Принцип: берём случайные точки из стандартного нормального
    распределения (латентное пространство) и пропускаем через декодер.
    """
    model.eval()
    with torch.no_grad():
        # Сэмплируем из латентного пространства
        z = torch.randn(n_samples, 2)
        generated = model.decode(z).numpy().flatten()

    # Возвращаем к исходному масштабу
    generated = generated * (data_max - data_min) + data_min
    return generated

# Генерируем выборки разного объёма
N_LARGE = 2000
N_MEDIUM = 500

generated_large = generate_vae(vae, N_LARGE, data_min, data_max)
generated_medium = generate_vae(vae, N_MEDIUM, data_min, data_max)

# Сохранение сгенерированных данных
np.save("vae_generated_n2000.npy", generated_large)
np.save("vae_generated_n500.npy", generated_medium)

# Также в CSV
pd.DataFrame({"value": generated_large}).to_csv("vae_generated_n2000.csv", index=False)
pd.DataFrame({"value": generated_medium}).to_csv("vae_generated_n500.csv", index=False)

print(f"\nСгенерировано {N_LARGE} значений (сохранено в vae_generated_n2000.csv)")
print(f"Сгенерировано {N_MEDIUM} значений (сохранено в vae_generated_n500.csv)")

# --- 2.6. Быстрый визуальный контроль ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data_norm, bins=8, density=True, alpha=0.6, label="Исходные (норм.)")
axes[0].set_title("Исходные данные (нормализованные)")
axes[0].legend()

axes[1].hist(generated_large, bins=30, density=True, alpha=0.6, label="VAE (n=2000)")
axes[1].set_title("Сгенерированные данные VAE")
axes[1].legend()

plt.tight_layout()
plt.savefig("vae_comparison.png", dpi=150)
plt.close()
print("Сравнение гистограмм сохранено в vae_comparison.png")


# ============================================================================
# РАЗДЕЛ 3. САМОСТОЯТЕЛЬНАЯ СТАТИСТИЧЕСКАЯ ОБРАБОТКА
# ============================================================================
# Внимание: все расчёты выполнять ВРУЧНУЮ по формулам.
# Не использовать scipy.stats.fit() для основных расчётов.

print("\n" + "=" * 70)
print("РАЗДЕЛ 3. Статистическая обработка сгенерированной выборки (n=2000)")
print("=" * 70)

data = generated_large  # работаем с большой выборкой
n = len(data)

# --- 3.1. Упорядоченная статистическая совокупность ---
data_sorted = np.sort(data)

# --- 3.2. Числовые характеристики (рассчитать по формулам!) ---

# <-- ЗАПОЛНИТЬ: математическое ожидание
# Формула: X̄ = (1/n) * Σ x_i
mean = ...

# <-- ЗАПОЛНИТЬ: дисперсия (несмещённая)
# Формула: D = (1/(n-1)) * Σ (x_i - X̄)²
variance = ...

# <-- ЗАПОЛНИТЬ: среднее квадратическое отклонение
# Формула: σ = √D
std = ...

# <-- ЗАПОЛНИТЬ: коэффициент вариации (%)
# Формула: V = (σ / X̄) * 100%
cv = ...

# <-- ЗАПОЛНИТЬ: медиана (через np.median или вручную)
median = ...

# <-- ЗАПОЛНИТЬ: мода (значение с максимальной частотой)
# Подсказка: используйте np.unique с return_counts
mode = ...

# <-- ЗАПОЛНИТЬ: асимметрия
# Формула: A_s = (1/n * Σ (x_i - X̄)³) / σ³
# Подсказка: можно через центральные моменты
asymmetry = ...

# <-- ЗАПОЛНИТЬ: эксцесс
# Формула: E_x = (1/n * Σ (x_i - X̄)⁴) / σ⁴ - 3
kurtosis = ...

print(f"\nЧисловые характеристики (n={n}):")
print(f"  Среднее:              {mean:.4f}")
print(f"  Дисперсия:            {variance:.4f}")
print(f"  СКО:                 {std:.4f}")
print(f"  Коэффициент вариации: {cv:.2f}%")
print(f"  Медиана:             {median:.4f}")
print(f"  Мода:                {mode}")
print(f"  Асимметрия:          {asymmetry:.4f}")
print(f"  Эксцесс:             {kurtosis:.4f}")


# --- 3.3. Группировка данных ---

# <-- ЗАПОЛНИТЬ: число интервалов по формуле Стерджеса
# K = 1 + 3.3 * lg(n)
K = ...

# <-- ЗАПОЛНИТЬ: величина интервала
# H = R / K, где R = x_max - x_min
H = ...

# <-- ЗАПОЛНИТЬ: границы интервалов
# Подсказка: np.linspace(x_min, x_max + H, K+1)
bins = ...

# <-- ЗАПОЛНИТЬ: расчёт частот, частостей, накопленных частот
# Используйте np.histogram или np.digitize
freq, bin_edges = ...  # <-- ЗАПОЛНИТЬ
midpoints = ...         # <-- ЗАПОЛНИТЬ: середины интервалов
freq_rel = ...          # <-- ЗАПОЛНИТЬ: частости p_i = n_i / n
freq_cum = ...          # <-- ЗАПОЛНИТЬ: накопленные частоты
freq_cum_rel = ...     # <-- ЗАПОЛНИТЬ: накопленные частости

# Сводная таблица
summary_table = pd.DataFrame({
    "№": range(1, K + 1),
    "Границы": [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(K)],
    "Середина": midpoints,
    "Частота": freq,
    "Частость": freq_rel,
    "Накопл. частота": freq_cum,
    "Накопл. частость": freq_cum_rel
})
print("\nСводная таблица:")
print(summary_table.to_string(index=False))
summary_table.to_csv("summary_table_n2000.csv", index=False)


# --- 3.4. Построение графиков ---

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Гистограмма
axes[0, 0].bar(midpoints, freq, width=H * 0.9, edgecolor='black', alpha=0.7)
axes[0, 0].set_title("Гистограмма")
axes[0, 0].set_xlabel("Значения")
axes[0, 0].set_ylabel("Частота")

# Полигон
axes[0, 1].plot(midpoints, freq, 'o-', color='blue')
axes[0, 1].set_title("Полигон")
axes[0, 1].set_xlabel("Середины интервалов")
axes[0, 1].set_ylabel("Частота")

# Ломаная кумулята
axes[1, 0].plot(bin_edges[:-1], freq_cum, 'o-', color='green')
axes[1, 0].set_title("Ломаная кумулята")
axes[1, 0].set_xlabel("Интервалы")
axes[1, 0].set_ylabel("Накопленные частоты")

# Ступенчатая кумулята
axes[1, 1].step(bin_edges[:-1], freq_cum, where='post', color='red')
axes[1, 1].set_title("Ступенчатая кумулята")
axes[1, 1].set_xlabel("Середины интервалов")
axes[1, 1].set_ylabel("Накопленные частоты")

plt.tight_layout()
plt.savefig("plots_n2000.png", dpi=150)
plt.close()
print("Графики сохранены в plots_n2000.png")


# ============================================================================
# РАЗДЕЛ 4. ИДЕНТИФИКАЦИЯ ЗАКОНА РАСПРЕДЕЛЕНИЯ
# ============================================================================
# Проверить НЕ МЕНЕЕ ЧЕТЫРЁХ гипотез:
#   1. Нормальный закон
#   2. Закон Вейбулла
#   3. Логнормальный закон
#   4. Гамма-распределение
#
# Для КАЖДОЙ гипотезы:
#   - оценить параметры (методом моментов или максимального правдоподобия)
#   - рассчитать теоретические частоты
#   - проверить согласие по ТРЕМ критериям:
#     а) Пирсона (хи-квадрат)
#     б) Колмогорова–Смирнова
#     в) Андерсона–Дарлинга

print("\n" + "=" * 70)
print("РАЗДЕЛ 4. Идентификация закона распределения (n=2000)")
print("=" * 70)

alpha = 0.05  # уровень значимости


# --- 4.1. Функция для расчёта критерия Пирсона ---

def chi_square_test(observed_freq, expected_freq, n_params):
    """
    Критерий согласия Пирсона (хи-квадрат).

    Параметры:
      observed_freq  — эмпирические частоты по интервалам
      expected_freq  — теоретические частоты по интервалам
      n_params       — число оценённых параметров распределения

    Возвращает:
      chi2_stat  — расчётное значение статистики
      dof       — число степеней свободы
      p_value   — p-значение
      verdict   — "принят" или "отклонён"
    """
    # <-- ЗАПОЛНИТЬ: рассчитать хи-квадрат
    # Формула: χ² = Σ (n_i - n_{Ti})² / n_{Ti}
    # Объедините интервалы с ожидаемой частотой < 5
    chi2_stat = ...
    dof = len(observed_freq) - n_params - 1
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    verdict = "принят" if p_value > alpha else "отклонён"
    return chi2_stat, dof, p_value, verdict


# --- 4.2. Функция для расчёта критерия Колмогорова–Смирнова ---

def ks_test(data, dist_cdf, *params):
    """
    Критерий согласия Колмогорова–Смирнова.

    Параметры:
      data     — выборка (не сгруппированная!)
      dist_cdf — функция распределения CDF (кумулята)
      *params  — параметры распределения

    Возвращает:
      ks_stat  — статистика D
      p_value  — p-значение
      verdict  — "принят" или "отклонён"
    """
    # <-- ЗАПОЛНИТЬ: рассчитать статистику Колмогорова
    # D = max |F_emp(x) - F_theor(x)|
    # Подсказка: используйте stats.kstest или реализуйте вручную
    ks_stat = ...
    p_value = ...
    verdict = "принят" if p_value > alpha else "отклонён"
    return ks_stat, p_value, verdict


# --- 4.3. Функция для расчёта критерия Андерсона–Дарлинга ---

def ad_test(data, dist_name, *params):
    """
    Критерий согласия Андерсона–Дарлинга.

    Более чувствителен к отклонениям на хвостах распределения,
    чем критерий Колмогорова–Смирнова.

    Параметры:
      data      — выборка
      dist_name — название распределения для scipy.stats
      *params   — параметры распределения

    Возвращает:
      ad_stat   — статистика A²
      verdict   — "принят" или "отклонён"
    """
    # <-- ЗАПОЛНИТЬ: рассчитать статистику Андерсона–Дарлинга
    # Подсказка: используйте stats.anderson или реализуйте по формуле
    # A² = -n - (1/n) * Σ (2i-1) * [ln(F(x_i)) + ln(1 - F(x_{n+1-i}))]
    ad_stat = ...
    # Сравните с критическим значением для alpha=0.05
    ad_crit = ...  # <-- ЗАПОЛНИТЬ: критическое значение
    verdict = "принят" if ad_stat < ad_crit else "отклонён"
    return ad_stat, verdict


# --- 4.4. Гипотеза 1: Нормальный закон ---

print("\n--- Гипотеза 1: Нормальный закон ---")

# <-- ЗАПОЛНИТЬ: оценка параметров методом моментов
# Для нормального: mu = X̄, sigma = σ
norm_mu = ...
norm_sigma = ...

# <-- ЗАПОЛНИТЬ: теоретические частоты
# n_Ti = n * [Φ(x_{i+1}) - Φ(x_i)]
# где Φ — функция нормального распределения с параметрами mu, sigma
norm_expected = ...  # <-- ЗАПОЛНИТЬ

# Проверка по трём критериям
chi2_norm, dof_norm, p_chi2_norm, verdict_chi2_norm = chi_square_test(
    freq, norm_expected, n_params=2
)
ks_norm, p_ks_norm, verdict_ks_norm = ks_test(
    data, stats.norm.cdf, norm_mu, norm_sigma
)
ad_norm, verdict_ad_norm = ad_test(data, 'norm', norm_mu, norm_sigma)

print(f"  Пирсон:        χ² = {chi2_norm:.2f}, p = {p_chi2_norm:.4f} — {verdict_chi2_norm}")
print(f"  Колмогоров:    D  = {ks_norm:.4f}, p = {p_ks_norm:.4f} — {verdict_ks_norm}")
print(f"  Андерсон:     A² = {ad_norm:.2f} — {verdict_ad_norm}")


# --- 4.5. Гипотеза 2: Закон Вейбулла ---

print("\n--- Гипотеза 2: Закон Вейбулла ---")

# <-- ЗАПОЛНИТЬ: оценка параметров методом моментов
# Для Вейбулла: оценить параметр формы k и масштаба λ
# Подсказка: используйте соотношения
#   σ/μ = [Γ(1+2/k)/Γ(1+1/k)² - 1]^(1/2)  — найти k
#   λ = μ / Γ(1+1/k)                       — найти λ
weibull_k = ...
weibull_lambda = ...

# <-- ЗАПОЛНИТЬ: теоретические частоты
weibull_expected = ...

# Проверка
chi2_weib, dof_weib, p_chi2_weib, verdict_chi2_weib = chi_square_test(
    freq, weibull_expected, n_params=2
)
ks_weib, p_ks_weib, verdict_ks_weib = ks_test(
    data, stats.weibull_min.cdf, weibull_k, scale=weibull_lambda
)
ad_weib, verdict_ad_weib = ad_test(data, 'weibull_min', weibull_k, scale=weibull_lambda)

print(f"  Пирсон:        χ² = {chi2_weib:.2f}, p = {p_chi2_weib:.4f} — {verdict_chi2_weib}")
print(f"  Колмогоров:    D  = {ks_weib:.4f}, p = {p_ks_weib:.4f} — {verdict_ks_weib}")
print(f"  Андерсон:     A² = {ad_weib:.2f} — {verdict_ad_weib}")


# --- 4.6. Гипотеза 3: Логнормальный закон ---

print("\n--- Гипотеза 3: Логнормальный закон ---")

# <-- ЗАПОПОЛНИТЬ: оценка параметров
# mu_log = mean(ln(x_i)), sigma_log = std(ln(x_i))
# (только если все x_i > 0)
lognorm_mu = ...
lognorm_sigma = ...

# <-- ЗАПОЛНИТЬ: теоретические частоты
lognorm_expected = ...

chi2_logn, dof_logn, p_chi2_logn, verdict_chi2_logn = chi_square_test(
    freq, lognorm_expected, n_params=2
)
ks_logn, p_ks_logn, verdict_ks_logn = ks_test(
    data, stats.lognorm.cdf, lognorm_sigma, scale=np.exp(lognorm_mu)
)
ad_logn, verdict_ad_logn = ad_test(data, 'lognorm', lognorm_sigma, scale=np.exp(lognorm_mu))

print(f"  Пирсон:        χ² = {chi2_logn:.2f}, p = {p_chi2_logn:.4f} — {verdict_chi2_logn}")
print(f"  Колмогоров:    D  = {ks_logn:.4f}, p = {p_ks_logn:.4f} — {verdict_ks_logn}")
print(f"  Андерсон:     A² = {ad_logn:.2f} — {verdict_ad_logn}")


# --- 4.7. Гипотеза 4: Гамма-распределение ---

print("\n--- Гипотеза 4: Гамма-распределение ---")

# <-- ЗАПОЛНИТЬ: оценка параметров методом моментов
# Для гамма: a = (μ/σ)², scale = σ²/μ
gamma_a = ...
gamma_scale = ...

# <-- ЗАПОЛНИТЬ: теоретические частоты
gamma_expected = ...

chi2_gam, dof_gam, p_chi2_gam, verdict_chi2_gam = chi_square_test(
    freq, gamma_expected, n_params=2
)
ks_gam, p_ks_gam, verdict_ks_gam = ks_test(
    data, stats.gamma.cdf, gamma_a, scale=gamma_scale
)
ad_gam, verdict_ad_gam = ad_test(data, 'gamma', gamma_a, scale=gamma_scale)

print(f"  Пирсон:        χ² = {chi2_gam:.2f}, p = {p_chi2_gam:.4f} — {verdict_chi2_gam}")
print(f"  Колмогоров:    D  = {ks_gam:.4f}, p = {p_ks_gam:.4f} — {verdict_ks_gam}")
print(f"  Андерсон:     A² = {ad_gam:.2f} — {verdict_ad_gam}")


# --- 4.8. Сводная таблица результатов ---

results_table = pd.DataFrame({
    "Закон": ["Нормальный", "Вейбулла", "Логнормальный", "Гамма"],
    "Пирсон χ²": [chi2_norm, chi2_weib, chi2_logn, chi2_gam],
    "Пирсон p": [p_chi2_norm, p_chi2_weib, p_chi2_logn, p_chi2_gam],
    "Пирсон вердикт": [verdict_chi2_norm, verdict_chi2_weib,
                        verdict_chi2_logn, verdict_chi2_gam],
    "КС D": [ks_norm, ks_weib, ks_logn, ks_gam],
    "КС p": [p_ks_norm, p_ks_weib, p_ks_logn, p_ks_gam],
    "КС вердикт": [verdict_ks_norm, verdict_ks_weib,
                    verdict_ks_logn, verdict_ks_gam],
    "АД A²": [ad_norm, ad_weib, ad_logn, ad_gam],
    "АД вердикт": [verdict_ad_norm, verdict_ad_weib,
                    verdict_ad_logn, verdict_ad_gam],
})

print("\n" + "=" * 70)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (n=2000)")
print("=" * 70)
print(results_table.to_string(index=False))
results_table.to_csv("hypothesis_results_n2000.csv", index=False)

# --- 4.9. Итоговый вывод ---
# <-- ЗАПОЛНИТЬ: проанализируйте таблицу и сделайте вывод
# Какой закон принят наибольшим числом критериев?
# Есть ли расхождения между критериями? Чем они объясняются?
#
# ВЫВОД: ...
final_conclusion = """
ВЫВОД (заполнить самостоятельно):
...
"""
print(final_conclusion)


# ============================================================================
# РАЗДЕЛ 5. ИССЛЕДОВАНИЕ УСТОЙЧИВОСТИ (n=500 vs n=2000)
# ============================================================================

print("\n" + "=" * 70)
print("РАЗДЕЛ 5. Исследование устойчивости идентификации")
print("=" * 70)

# --- 5.1. Повторить обработку для n=500 ---
# <-- ЗАПОЛНИТЬ: выполните те же шаги (3.2–4.9) для generated_medium
# Сохраните результаты отдельно

# Скопируйте и адаптируйте код из разделов 3–4 для n=500
# Заполните таблицу результатов аналогично results_table

# <-- ЗАПОЛНИТЬ: results_table_n500 = ...

# --- 5.2. Сравнение ---
# <-- ЗАПОПОЛНИТЬ: заполните сравнительную таблицу
comparison = pd.DataFrame({
    "Параметр": ["Среднее", "СКО", "Коэф. вариации", "Асимметрия",
                 "Эксцесс", "Идентиф. закон (Пирсон)",
                 "Идентиф. закон (КС)", "Идентиф. закон (АД)"],
    "n=500": [...],   # <-- ЗАПОЛНИТЬ
    "n=2000": [...],  # <-- ЗАПОЛНИТЬ
    "Разница": [...]   # <-- ЗАПОЛНИТЬ
})
print(comparison.to_string(index=False))
comparison.to_csv("stability_comparison.csv", index=False)

# --- 5.3. График зависимости статистик критериев от объёма выборки ---
# <-- ЗАПОЛНИТЬ: постройте график
# По оси X: n = [500, 1000, 1500, 2000]
# По оси Y: значения χ², D, A² для каждого закона
# (для этого нужно сгенерировать выборки нескольких объёмов)

fig, ax = plt.subplots(figsize=(10, 6))
# <-- ЗАПОЛНИТЬ: добавьте линии для каждого закона и каждого критерия
ax.set_title("Зависимость статистик критериев от объёма выборки")
ax.set_xlabel("Объём выборки n")
ax.set_ylabel("Значение статистики")
ax.legend()
plt.tight_layout()
plt.savefig("stability_plot.png", dpi=150)
plt.close()


# ============================================================================
# РАЗДЕЛ 6. ВАЛИДАЦИЯ ГЕНЕРАТОРА
# ============================================================================

print("\n" + "=" * 70)
print("РАЗДЕЛ 6. Валидация генератора")
print("=" * 70)

# --- 6.1. Сравнение характеристик исходной и сгенерированной выборок ---

validation = pd.DataFrame({
    "Параметр": ["Среднее", "СКО", "Медиана", "Минимум",
                 "Максимум", "Коэф. вариации (%)", "Асимметрия", "Эксцесс"],
    "Исходная (n=50)": [mean_orig, std_orig, median_orig, min_orig,
                        max_orig, cv_orig, ...],  # <-- ЗАПОЛНИТЬ асимметрию, эксцесс исходной
    "VAE (n=2000)": [mean, std, median, data.min(),
                     data.max(), cv, asymmetry, kurtosis],
    "Отклонение (%)": [...]  # <-- ЗАПОЛНИТЬ: относительное отклонение
})
print(validation.to_string(index=False))
validation.to_csv("generator_validation.csv", index=False)

# --- 6.2. Вывод о качестве генератора ---
# <-- ЗАПОЛНИТЬ: проанализируйте, сохраняет ли VAE:
#   - среднее значение?
#   - разброс (СКО)?
#   - форму распределения (асимметрия, эксцесс)?
#   - какие характеристики искажаются и почему?
generator_conclusion = """
ВЫВОД О КАЧЕСТВЕ ГЕНЕРАТОРА (заполнить):
...
"""
print(generator_conclusion)


# ============================================================================
# РАЗДЕЛ 7. ВЕРИФИКАЦИЯ (самопроверка)
# ============================================================================
# Здесь РАЗРЕШАЕТСЯ использовать готовые функции для проверки
# правильности ручных расчётов.

print("\n" + "=" * 70)
print("РАЗДЕЛ 7. Верификация (самопроверка)")
print("=" * 70)

# <-- ЗАПОЛНИТЬ: сравните ваши ручные оценки параметров
# с результатами scipy.stats.fit

# Нормальный
norm_fit = stats.norm.fit(data)
print(f"Нормальный: ручные = ({norm_mu:.4f}, {norm_sigma:.4f}), "
      f"scipy = ({norm_fit[0]:.4f}, {norm_fit[1]:.4f})")

# Вейбулл
weibull_fit = stats.weibull_min.fit(data, floc=0)
print(f"Вейбулл: ручные = ({weibull_k:.4f}, {weibull_lambda:.4f}), "
      f"scipy = ({weibull_fit[0]:.4f}, {weibull_fit[2]:.4f})")

# Логнормальный
logn_fit = stats.lognorm.fit(data, floc=0)
print(f"Логнормальный: ручные = ({lognorm_mu:.4f}, {lognorm_sigma:.4f}), "
      f"scipy = ({logn_fit[0]:.4f}, {np.log(logn_fit[2]):.4f})")

# Гамма
gamma_fit = stats.gamma.fit(data, floc=0)
print(f"Гамма: ручные = ({gamma_a:.4f}, {gamma_scale:.4f}), "
      f"scipy = ({gamma_fit[0]:.4f}, {gamma_fit[2]:.4f})")

# <-- ЗАПОЛНИТЬ: сделайте вывод о точности ручных оценок
verification_conclusion = """
ВЫВОД О ТОЧНОСТИ РУЧНЫХ РАСЧЁТОВ (заполнить):
...
"""
print(verification_conclusion)


# ============================================================================
# КОНЕЦ
# ============================================================================
print("\n" + "=" * 70)
print("Работа завершена. Проверьте созданные файлы:")
print("  - hist_original.png")
print("  - vae_comparison.png")
print("  - vae_generated_n2000.csv / .npy")
print("  - vae_generated_n500.csv / .npy")
print("  - summary_table_n2000.csv")
print("  - plots_n2000.png")
print("  - hypothesis_results_n2000.csv")
print("  - stability_comparison.csv")
print("  - stability_plot.png")
print("  - generator_validation.csv")
print("=" * 70)
