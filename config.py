"""
Конфигурационный файл для анализа данных антикроссинга мод ФМР и резонатора
"""

import os
import numpy as np

# =============================================================================
# ПУТИ К ФАЙЛАМ И ПАПКАМ
# =============================================================================

# Базовая папка проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка с экспериментальными данными
DATA_DIR = os.path.join(BASE_DIR, "data")

# Папка для сохранения результатов
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Файлы с данными
DATA_FILES = {
    'S21': 'CoherentCoupling_S21.txt',
    'S12': 'CoherentCoupling_S12.txt'
}

# =============================================================================
# ПАРАМЕТРЫ АНАЛИЗА
# =============================================================================

# Тип анализируемых данных ('S21' или 'S12')
ANALYSIS_TYPE = 'S21'  # Можно изменить на 'S12'

# Количество мод ФМР для анализа (автоматическое определение если None)
NUM_MODES = None

# Диапазон частот для анализа (ГГц) - None для использования всего диапазона
FREQUENCY_RANGE = (3.4, 3.9)  # Например: (8.0, 12.0)

# Диапазон полей для анализа (Э) - None для использования всего диапазона
FIELD_RANGE = (2875, 3040)  # Например: (1000, 5000)

# Параметры обработки данных
IGNORE_LAST_ROW = True  # Игнорировать последнюю строку данных (обычно содержит ошибочные данные)
MIN_FIELD_THRESHOLD = 10  # Минимальное значение поля для валидных данных (Э)

# Параметры расчета частоты ФМР
GYROMAGNETIC_RATIO = 2.8e-3  # Гиромагнитное отношение (ГГц/Э)
ANISOTROPY_FIELD = 0  # Поле анизотропии (Э) - сдвиг частоты ФМР от нуля

# =============================================================================
# ПАРАМЕТРЫ ФИТИНГА
# =============================================================================

# Начальные параметры для одной моды ФМР
# [kappa, beta, wc, wm, J, Gamma, alpha, gamma]
INITIAL_PARAMS_SINGLE = {
    'kappa': 0.1,      # внешние потери резонатора (ГГц)
    'beta': 0.05,      # внутренние потери резонатора (ГГц)
    'wc': 10.0,        # резонансная частота резонатора (ГГц)
    'wm': 10.0,        # резонансная частота магнонной моды (ГГц)
    'J': 0.1,          # коэффициент когерентной связи (ГГц)
    'Gamma': 0.05,     # коэффициент диссипативной связи (ГГц)
    'alpha': 0.02,     # внутренние потери магнонной моды (ГГц)
    'gamma': 0.05      # внешние потери магнонной моды (ГГц)
}

# Границы параметров для оптимизации [min, max]
PARAM_BOUNDS = {
    'kappa': (0.001, 1.0),
    'beta': (0.001, 0.5),
    'wc': (5.0, 15.0),
    'wm': (5.0, 15.0),
    'J': (0.001, 1.0),
    'Gamma': (0.001, 0.5),
    'alpha': (0.001, 0.2),
    'gamma': (0.001, 0.5)
}

# =============================================================================
# ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ
# =============================================================================

# Размер фигур для графиков
FIGURE_SIZE = (12, 8)

# DPI для сохранения графиков
FIGURE_DPI = 300

# Формат сохранения графиков
FIGURE_FORMAT = 'png'

# Цветовая карта для контурных графиков
COLORMAP = 'viridis'

# Количество уровней в контурном графике
CONTOUR_LEVELS = 50

# =============================================================================
# ПАРАМЕТРЫ ЛОГИРОВАНИЯ
# =============================================================================

# Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR')
LOG_LEVEL = 'INFO'

# Имя файла лога
LOG_FILE = 'anticrossing_analysis.log'

# =============================================================================
# ТЕОРЕТИЧЕСКАЯ МОДЕЛЬ
# =============================================================================

def theoretical_model(freq, field, params, analysis_type='S21'):
    """
    Теоретическая модель для S-параметров с учетом связи мод
    
    Parameters:
    -----------
    freq : array_like
        Частоты (ГГц)
    field : float
        Магнитное поле (Э)
    params : dict
        Параметры модели
    analysis_type : str
        Тип анализа ('S21' или 'S12')
    
    Returns:
    --------
    S_param : complex array
        Вычисленные S-параметры
    """
    
    omega = 2 * np.pi * freq  # Угловая частота
    
    # Извлечение параметров
    kappa = params.get('kappa', INITIAL_PARAMS_SINGLE['kappa'])
    beta = params.get('beta', INITIAL_PARAMS_SINGLE['beta'])
    wc = 2 * np.pi * params.get('wc', INITIAL_PARAMS_SINGLE['wc'])
    wm = 2 * np.pi * params.get('wm', INITIAL_PARAMS_SINGLE['wm'])
    J = params.get('J', INITIAL_PARAMS_SINGLE['J'])
    Gamma = params.get('Gamma', INITIAL_PARAMS_SINGLE['Gamma'])
    alpha = params.get('alpha', INITIAL_PARAMS_SINGLE['alpha'])
    gamma = params.get('gamma', INITIAL_PARAMS_SINGLE['gamma'])
    
    # Фазовый множитель в зависимости от типа измерения
    theta = 0 if analysis_type == 'S21' else np.pi
    
    # Вычисление S-параметра
    coupling_term = (1j * J + Gamma * np.exp(1j * theta))**2
    magnon_denominator = 1j * (omega - wm) - (alpha + gamma)
    cavity_part = 1j * (omega - wc) - (kappa + beta)
    
    S_param = 1 + kappa / (cavity_part - coupling_term / magnon_denominator)
    
    return S_param

def calculate_magnon_frequency(field, gamma_factor=None, h_anisotropy=None):
    """
    Вычисление частоты магнонной моды в зависимости от поля
    
    Parameters:
    -----------
    field : float
        Магнитное поле (Э)
    gamma_factor : float, optional
        Гиромагнитное отношение (ГГц/Э). Если None, используется GYROMAGNETIC_RATIO
    h_anisotropy : float, optional
        Поле анизотропии (Э). Если None, используется ANISOTROPY_FIELD
        
    Returns:
    --------
    freq : float
        Частота магнонной моды (ГГц)
    """
    if gamma_factor is None:
        gamma_factor = GYROMAGNETIC_RATIO
    if h_anisotropy is None:
        h_anisotropy = ANISOTROPY_FIELD
    
    return gamma_factor * (field + h_anisotropy)

# =============================================================================
# ДОПОЛНИТЕЛЬНЫЕ НАСТРОЙКИ
# =============================================================================

# Создание папки результатов если её нет
os.makedirs(RESULTS_DIR, exist_ok=True)

# Максимальное количество итераций для оптимизации
MAX_ITERATIONS = 1000

# Допуск для сходимости оптимизации
OPTIMIZATION_TOLERANCE = 1e-8

# Метод оптимизации
OPTIMIZATION_METHOD = 'leastsq'  # 'leastsq', 'least_squares', 'minimize'