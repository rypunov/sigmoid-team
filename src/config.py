from pathlib import Path

# Пути проекта
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIG_DIR = ARTIFACTS_DIR / "figures"
REPORT_DIR = ARTIFACTS_DIR / "reports"
MODEL_DIR = ARTIFACTS_DIR / "models"
CACHE_DIR = ARTIFACTS_DIR / "cache"

# Создаем папки, если их нет
for path in [FIG_DIR, REPORT_DIR, MODEL_DIR, CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Настройки данных
N_FILES = 20  # Количество областей (Obl01..Obl20)
N_RHO_RANGES = 4
N_T_RANGES = 5

# Единицы измерения (фиксируем для всей команды)
RHO_UNITS = "g/cm3"
T_UNITS = "K"
P_UNITS = "GPa"
E_UNITS = "kJ/g"  # Внимание: в ТЗ просят эВ/атом, но в данных kJ/g. Обсудим позже.

# Параметры нормализации
# P и rho меняются на много порядков -> используем log10
USE_LOG_RHO = True
USE_LOG_P = True
# E имеет отрицательные значения -> используем MinMax или StandardScaler