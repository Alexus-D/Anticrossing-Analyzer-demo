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

# Калибровочная точка для расчета частоты ФМР (более удобно чем поле анизотропии)
FMR_CALIBRATION_FIELD = 3000  # Поле, при котором известна резонансная частота (Э)
FMR_CALIBRATION_FREQUENCY = 8.4  # Резонансная частота при калибровочном поле (ГГц)

# =============================================================================
# ПАРАМЕТРЫ ФИТИНГА
# =============================================================================

# Параметры ширины линий (более удобные для настройки)
CAVITY_LINEWIDTH = 1.0      # Полная ширина линии резонатора (ГГц)
MAGNON_LINEWIDTH = 0.14     # Полная ширина линии магнонной моды (ГГц)

# Начальные параметры для одной моды ФМР
INITIAL_PARAMS_SINGLE = {
    'wc': 3.65,        # резонансная частота резонатора (ГГц)
    'wm': 10.0,        # резонансная частота магнонной моды (ГГц)
    'J': 0.1,          # коэффициент когерентной связи (ГГц)
    'Gamma': 0.05,     # коэффициент диссипативной связи (ГГц)
    'cavity_loss': CAVITY_LINEWIDTH / 2,  # полная ширина линии резонатора / 2 (ГГц)
    'magnon_loss': MAGNON_LINEWIDTH / 2   # полная ширина линии магнонов / 2 (ГГц)
}

# Границы параметров для оптимизации [min, max]
PARAM_BOUNDS = {
    'wc': (3.0, 4.0),
    'wm': (5.0, 15.0),
    'J': (0.001, 1.0),
    'Gamma': (0.001, 0.5),
    'cavity_loss': (0.001, 2.0),
    'magnon_loss': (0.001, 1.0)
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
    wc = 2 * np.pi * params.get('wc', INITIAL_PARAMS_SINGLE['wc'])
    wm = 2 * np.pi * params.get('wm', INITIAL_PARAMS_SINGLE['wm'])
    J = params.get('J', INITIAL_PARAMS_SINGLE['J'])
    Gamma = params.get('Gamma', INITIAL_PARAMS_SINGLE['Gamma'])
    cavity_loss = params.get('cavity_loss', INITIAL_PARAMS_SINGLE['cavity_loss'])
    magnon_loss = params.get('magnon_loss', INITIAL_PARAMS_SINGLE['magnon_loss'])
    
    # Фазовый множитель в зависимости от типа измерения
    theta = 0 if analysis_type == 'S21' else np.pi
    
    # Вычисление S-параметра
    # Упрощенная модель: используем полную ширину линии вместо разделения на внутренние/внешние потери
    coupling_term = (1j * J + Gamma * np.exp(1j * theta))**2
    magnon_denominator = 1j * (omega - wm) - magnon_loss
    cavity_part = 1j * (omega - wc) - cavity_loss
    
    S_param = 1 + cavity_loss / (cavity_part - coupling_term / magnon_denominator)
    
    return S_param

def calculate_magnon_frequency(field, gamma_factor=None, calib_field=None, calib_freq=None):
    """
    Вычисление частоты магнонной моды в зависимости от поля
    Использует калибровочную точку для более удобной настройки
    
    Parameters:
    -----------
    field : float
        Магнитное поле (Э)
    gamma_factor : float, optional
        Гиромагнитное отношение (ГГц/Э). Если None, используется GYROMAGNETIC_RATIO
    calib_field : float, optional
        Калибровочное поле (Э). Если None, используется FMR_CALIBRATION_FIELD
    calib_freq : float, optional
        Частота при калибровочном поле (ГГц). Если None, используется FMR_CALIBRATION_FREQUENCY
        
    Returns:
    --------
    freq : float
        Частота магнонной моды (ГГц)
    """
    if gamma_factor is None:
        gamma_factor = GYROMAGNETIC_RATIO
    if calib_field is None:
        calib_field = FMR_CALIBRATION_FIELD
    if calib_freq is None:
        calib_freq = FMR_CALIBRATION_FREQUENCY
    
    # Рассчитываем частоту через калибровочную точку
    # f = f_calib + γ * (H - H_calib)
    return calib_freq + gamma_factor * (field - calib_field)

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