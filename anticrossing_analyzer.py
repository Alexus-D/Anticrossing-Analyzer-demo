"""
Основной скрипт для анализа данных антикроссинга мод ФМР и резонатора

Автор: GitHub Copilot
Дата создания: 2025-10-06
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit, leastsq
import warnings
warnings.filterwarnings('ignore')

# Импорт конфигурации
import config

# =============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =============================================================================

def setup_logging():
    """Настройка системы логирования"""
    log_file = os.path.join(config.RESULTS_DIR, config.LOG_FILE)
    
    # Настройка форматирования
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настройка обработчиков
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
    
    # Конфигурация логгера
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        handlers=handlers,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)

# =============================================================================
# ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ
# =============================================================================

def load_data(file_path):
    """
    Загрузка экспериментальных данных из файла
    
    Parameters:
    -----------
    file_path : str
        Путь к файлу с данными
        
    Returns:
    --------
    frequencies : array
        Массив частот (ГГц)
    fields : array
        Массив магнитных полей (Э)
    s_params : 2D array
        Матрица S-параметров (дБ)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Загрузка данных
        data = np.loadtxt(file_path, delimiter='\t')
        
        # Извлечение частот (первая строка, кроме первого элемента)
        frequencies = data[0, 1:]
        
        # Извлечение полей (первый столбец, кроме первого элемента)  
        fields = data[1:, 0]
        
        # Извлечение S-параметров (остальная матрица)
        s_params = data[1:, 1:]
        
        # Обработка проблемных данных
        if config.IGNORE_LAST_ROW and len(fields) > 1:
            # Проверяем последнюю строку на проблемы (например, поле близко к нулю)
            if fields[-1] < config.MIN_FIELD_THRESHOLD:
                logger.warning(f"Игнорируем последнюю строку с подозрительным полем: {fields[-1]:.1f} Э")
                fields = fields[:-1]
                s_params = s_params[:-1, :]
        
        # Дополнительная фильтрация по минимальному полю
        valid_field_mask = fields >= config.MIN_FIELD_THRESHOLD
        if not np.all(valid_field_mask):
            invalid_count = np.sum(~valid_field_mask)
            logger.warning(f"Удаляем {invalid_count} строк с полем ниже {config.MIN_FIELD_THRESHOLD} Э")
            fields = fields[valid_field_mask]
            s_params = s_params[valid_field_mask, :]
        
        logger.info(f"Данные загружены: {len(frequencies)} частот, {len(fields)} полей")
        logger.info(f"Диапазон частот: {frequencies.min():.2f} - {frequencies.max():.2f} ГГц")
        logger.info(f"Диапазон полей: {fields.min():.0f} - {fields.max():.0f} Э")
        
        return frequencies, fields, s_params
        
    except Exception as e:
        logger.error(f"Ошибка загрузки данных из {file_path}: {e}")
        raise

def preprocess_data(frequencies, fields, s_params):
    """
    Предварительная обработка данных
    
    Parameters:
    -----------
    frequencies : array
        Массив частот
    fields : array
        Массив полей
    s_params : 2D array
        Матрица S-параметров в дБ
        
    Returns:
    --------
    freq_filtered : array
        Отфильтрованные частоты
    field_filtered : array
        Отфильтрованные поля
    s_filtered : 2D array
        Отфильтрованные S-параметры в линейном масштабе
    """
    logger = logging.getLogger(__name__)
    
    # Применение фильтров по частоте
    if config.FREQUENCY_RANGE is not None:
        freq_mask = (frequencies >= config.FREQUENCY_RANGE[0]) & (frequencies <= config.FREQUENCY_RANGE[1])
        frequencies = frequencies[freq_mask]
        s_params = s_params[:, freq_mask]
        logger.info(f"Применен фильтр по частоте: {config.FREQUENCY_RANGE}")
    
    # Применение фильтров по полю
    if config.FIELD_RANGE is not None:
        field_mask = (fields >= config.FIELD_RANGE[0]) & (fields <= config.FIELD_RANGE[1])
        fields = fields[field_mask]
        s_params = s_params[field_mask, :]
        logger.info(f"Применен фильтр по полю: {config.FIELD_RANGE}")
    
    # Преобразование из дБ в линейный масштаб
    s_linear = 10**(s_params / 20)
    
    logger.info("Предварительная обработка данных завершена")
    
    return frequencies, fields, s_linear

# =============================================================================
# ФИТИНГ ТЕОРЕТИЧЕСКОЙ МОДЕЛИ
# =============================================================================

def fit_spectrum(frequencies, s_spectrum, field_value, initial_params=None):
    """
    Подгонка теоретической модели к одному спектру
    
    Parameters:
    -----------
    frequencies : array
        Частоты для спектра
    s_spectrum : array
        Измеренные S-параметры (линейный масштаб)
    field_value : float
        Значение магнитного поля
    initial_params : dict
        Начальные параметры (если None, используются из config)
        
    Returns:
    --------
    fitted_params : dict
        Подогнанные параметры
    fitted_spectrum : array
        Теоретический спектр с подогнанными параметрами
    fit_quality : float
        Качество подгонки (R²)
    """
    logger = logging.getLogger(__name__)
    
    if initial_params is None:
        initial_params = config.INITIAL_PARAMS_SINGLE.copy()
    
    # Начальная оценка резонансной частоты магнонной моды
    initial_params['wm'] = config.calculate_magnon_frequency(field_value)
    
    # Функция для оптимизации
    def residuals(params_array):
        # Преобразование массива параметров в словарь
        param_names = ['wc', 'wm', 'J', 'Gamma', 'cavity_loss', 'magnon_loss']
        params_dict = dict(zip(param_names, params_array))
        
        # Вычисление теоретического спектра
        theory = config.theoretical_model(frequencies, field_value, params_dict, config.ANALYSIS_TYPE)
        
        # Извлечение амплитуды
        theory_amp = np.abs(theory)
        
        # Остатки
        return s_spectrum - theory_amp
    
    try:
        # Начальные значения параметров как массив
        param_names = ['wc', 'wm', 'J', 'Gamma', 'cavity_loss', 'magnon_loss']
        initial_values = [initial_params[name] for name in param_names]
        
        # Оптимизация
        result = leastsq(residuals, initial_values, maxfev=config.MAX_ITERATIONS)
        fitted_values = result[0]
        
        # Преобразование результата в словарь
        fitted_params = dict(zip(param_names, fitted_values))
        
        # Вычисление подогнанного спектра
        fitted_spectrum = np.abs(config.theoretical_model(
            frequencies, field_value, fitted_params, config.ANALYSIS_TYPE
        ))
        
        # Оценка качества подгонки (R²)
        ss_res = np.sum((s_spectrum - fitted_spectrum) ** 2)
        ss_tot = np.sum((s_spectrum - np.mean(s_spectrum)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return fitted_params, fitted_spectrum, r_squared
        
    except Exception as e:
        logger.warning(f"Ошибка фитинга для поля {field_value:.0f} Э: {e}")
        # Возврат исходных параметров в случае ошибки
        fitted_spectrum = np.abs(config.theoretical_model(
            frequencies, field_value, initial_params, config.ANALYSIS_TYPE
        ))
        return initial_params, fitted_spectrum, 0.0

def analyze_all_spectra(frequencies, fields, s_params_matrix):
    """
    Анализ всех спектров в данных
    
    Parameters:
    -----------
    frequencies : array
        Массив частот
    fields : array  
        Массив полей
    s_params_matrix : 2D array
        Матрица S-параметров
        
    Returns:
    --------
    results_df : DataFrame
        Результаты анализа для всех полей
    fitted_matrix : 2D array
        Матрица подогнанных S-параметров
    """
    logger = logging.getLogger(__name__)
    
    results = []
    fitted_matrix = np.zeros_like(s_params_matrix)
    
    logger.info(f"Начинаем анализ {len(fields)} спектров...")
    
    for i, field in enumerate(fields):
        spectrum = s_params_matrix[i, :]
        
        # Подгонка модели
        fitted_params, fitted_spectrum, r_squared = fit_spectrum(
            frequencies, spectrum, field
        )
        
        # Сохранение результатов
        result = fitted_params.copy()
        result['field'] = field
        result['r_squared'] = r_squared
        results.append(result)
        
        # Сохранение подогнанного спектра
        fitted_matrix[i, :] = fitted_spectrum
        
        # Прогресс
        if (i + 1) % max(1, len(fields) // 10) == 0:
            logger.info(f"Обработано {i + 1}/{len(fields)} спектров")
    
    results_df = pd.DataFrame(results)
    logger.info("Анализ спектров завершен")
    
    return results_df, fitted_matrix

# =============================================================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

def plot_contour_data(frequencies, fields, data_matrix, title, filename, 
                     vmin=None, vmax=None, log_scale=False):
    """
    Создание контурного графика данных
    
    Parameters:
    -----------
    frequencies : array
        Массив частот
    fields : array
        Массив полей
    data_matrix : 2D array
        Матрица данных для отображения
    title : str
        Заголовок графика
    filename : str
        Имя файла для сохранения
    vmin, vmax : float
        Пределы цветовой шкалы
    log_scale : bool
        Использовать логарифмическую шкалу
    """
    logger = logging.getLogger(__name__)
    
    plt.figure(figsize=config.FIGURE_SIZE)
    
    # Создание сетки координат
    F, H = np.meshgrid(frequencies, fields)
    
    # Параметры нормализации
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else None
    
    # Контурный график
    contour = plt.contourf(F, H, data_matrix, levels=config.CONTOUR_LEVELS, 
                          cmap=config.COLORMAP, norm=norm, vmin=vmin, vmax=vmax)
    
    # Оформление
    plt.colorbar(contour, label='Амплитуда S-параметра')
    plt.xlabel('Частота (ГГц)')
    plt.ylabel('Магнитное поле (Э)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Сохранение
    save_path = os.path.join(config.RESULTS_DIR, f"{filename}.{config.FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"График сохранен: {save_path}")

def plot_parameters_vs_field(results_df):
    """
    Построение графиков параметров в зависимости от поля
    
    Parameters:
    -----------
    results_df : DataFrame
        Результаты анализа
    """
    logger = logging.getLogger(__name__)
    
    # Параметры для отображения
    params_to_plot = [
        ('wc', 'Частота резонатора (ГГц)'),
        ('wm', 'Частота магнонной моды (ГГц)'),
        ('J', 'Когерентная связь J (ГГц)'),
        ('Gamma', 'Диссипативная связь Γ (ГГц)'),
        ('cavity_loss', 'Ширина линии резонатора (ГГц)'),
        ('magnon_loss', 'Ширина линии магнонов (ГГц)')
    ]
    
    # Создание субплотов
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (param, label) in enumerate(params_to_plot):
        ax = axes[i]
        
        # График параметра
        ax.plot(results_df['field'], results_df[param], 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Магнитное поле (Э)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} vs Поле')
    
    plt.tight_layout()
    
    # Сохранение
    save_path = os.path.join(config.RESULTS_DIR, f"parameters_vs_field.{config.FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"График параметров сохранен: {save_path}")

def plot_fit_quality(results_df):
    """
    График качества подгонки
    
    Parameters:
    -----------
    results_df : DataFrame
        Результаты анализа
    """
    logger = logging.getLogger(__name__)
    
    plt.figure(figsize=config.FIGURE_SIZE)
    
    plt.plot(results_df['field'], results_df['r_squared'], 'o-', linewidth=2, markersize=4)
    plt.xlabel('Магнитное поле (Э)')
    plt.ylabel('R² (качество подгонки)')
    plt.title('Качество подгонки теоретической модели')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Статистика
    mean_r2 = results_df['r_squared'].mean()
    plt.axhline(mean_r2, color='red', linestyle='--', alpha=0.7, 
                label=f'Среднее R² = {mean_r2:.3f}')
    plt.legend()
    
    # Сохранение
    save_path = os.path.join(config.RESULTS_DIR, f"fit_quality.{config.FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"График качества подгонки сохранен: {save_path}")

def plot_anticrossing_analysis(frequencies, fields, results_df):
    """
    Специальный график для анализа антикроссинга
    
    Parameters:
    -----------
    frequencies : array
        Массив частот
    fields : array
        Массив полей  
    results_df : DataFrame
        Результаты анализа
    """
    logger = logging.getLogger(__name__)
    
    plt.figure(figsize=config.FIGURE_SIZE)
    
    # График резонансных частот
    plt.plot(results_df['field'], results_df['wc'], 'b-', linewidth=2, 
             label='Частота резонатора $\\omega_c$')
    plt.plot(results_df['field'], results_df['wm'], 'r-', linewidth=2, 
             label='Частота магнонной моды $\\omega_m$')
    
    # Оформление
    plt.xlabel('Магнитное поле (Э)')
    plt.ylabel('Частота (ГГц)')
    plt.title('Анализ антикроссинга мод')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавление информации о связи
    textstr = f'Средняя когерентная связь J = {results_df["J"].mean():.3f} ГГц\n'
    textstr += f'Средняя диссипативная связь Γ = {results_df["Gamma"].mean():.3f} ГГц'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Сохранение
    save_path = os.path.join(config.RESULTS_DIR, f"anticrossing_analysis.{config.FIGURE_FORMAT}")
    plt.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')
    plt.close()
    
    logger.info(f"График антикроссинга сохранен: {save_path}")

# =============================================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

def save_results(results_df, frequencies, fields, original_data, fitted_data):
    """
    Сохранение всех результатов анализа
    
    Parameters:
    -----------
    results_df : DataFrame
        Результаты анализа параметров
    frequencies : array
        Массив частот
    fields : array
        Массив полей
    original_data : 2D array
        Исходные данные
    fitted_data : 2D array  
        Подогнанные данные
    """
    logger = logging.getLogger(__name__)
    
    # Сохранение параметров в CSV
    csv_path = os.path.join(config.RESULTS_DIR, 'extracted_parameters.csv')
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"Параметры сохранены в CSV: {csv_path}")
    
    # Сохранение исходных и подогнанных данных
    original_path = os.path.join(config.RESULTS_DIR, 'original_data.csv')
    fitted_path = os.path.join(config.RESULTS_DIR, 'fitted_data.csv')
    
    # Создание DataFrame для данных
    freq_df = pd.DataFrame(original_data, index=fields, columns=frequencies)
    freq_df.to_csv(original_path, encoding='utf-8')
    
    fitted_df = pd.DataFrame(fitted_data, index=fields, columns=frequencies)
    fitted_df.to_csv(fitted_path, encoding='utf-8')
    
    logger.info(f"Исходные данные сохранены: {original_path}")
    logger.info(f"Подогнанные данные сохранены: {fitted_path}")
    
    # Сохранение статистики
    stats_path = os.path.join(config.RESULTS_DIR, 'analysis_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"СТАТИСТИКА АНАЛИЗА АНТИКРОССИНГА МОД\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Тип анализа: {config.ANALYSIS_TYPE}\n")
        f.write(f"Количество спектров: {len(results_df)}\n")
        f.write(f"Диапазон частот: {frequencies.min():.2f} - {frequencies.max():.2f} ГГц\n")
        f.write(f"Диапазон полей: {fields.min():.0f} - {fields.max():.0f} Э\n\n")
        
        f.write(f"СРЕДНИЕ ЗНАЧЕНИЯ ПАРАМЕТРОВ:\n")
        f.write(f"{'-'*30}\n")
        for param in ['wc', 'wm', 'J', 'Gamma', 'cavity_loss', 'magnon_loss']:
            mean_val = results_df[param].mean()
            std_val = results_df[param].std()
            f.write(f"{param:12s}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        f.write(f"\nКАЧЕСТВО ПОДГОНКИ:\n")
        f.write(f"{'-'*20}\n")
        f.write(f"Среднее R²: {results_df['r_squared'].mean():.4f}\n")
        f.write(f"Минимальное R²: {results_df['r_squared'].min():.4f}\n")
        f.write(f"Максимальное R²: {results_df['r_squared'].max():.4f}\n")
    
    logger.info(f"Статистика сохранена: {stats_path}")

# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    """Главная функция выполнения анализа"""
    
    # Настройка логирования
    logger = setup_logging()
    logger.info("="*60)
    logger.info("НАЧАЛО АНАЛИЗА ДАННЫХ АНТИКРОССИНГА МОД")
    logger.info("="*60)
    
    try:
        # Проверка существования файла данных
        data_file = config.DATA_FILES[config.ANALYSIS_TYPE]
        data_path = os.path.join(config.DATA_DIR, data_file)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")
        
        logger.info(f"Анализируем данные: {data_file}")
        
        # Загрузка данных
        logger.info("Загрузка экспериментальных данных...")
        frequencies, fields, s_params = load_data(data_path)
        
        # Предварительная обработка
        logger.info("Предварительная обработка данных...")
        frequencies, fields, s_linear = preprocess_data(frequencies, fields, s_params)
        
        # Создание контурного графика исходных данных
        logger.info("Создание графика исходных данных...")
        plot_contour_data(
            frequencies, fields, s_linear,
            f'Исходные данные {config.ANALYSIS_TYPE}',
            'original_data_contour'
        )
        
        # Анализ всех спектров
        logger.info("Выполнение анализа спектров...")
        results_df, fitted_matrix = analyze_all_spectra(frequencies, fields, s_linear)
        
        # Создание контурного графика подогнанных данных
        logger.info("Создание графика подогнанных данных...")
        plot_contour_data(
            frequencies, fields, fitted_matrix,
            f'Подогнанные данные {config.ANALYSIS_TYPE}',
            'fitted_data_contour'
        )
        
        # Создание остальных графиков
        logger.info("Создание графиков результатов...")
        plot_parameters_vs_field(results_df)
        plot_fit_quality(results_df)
        plot_anticrossing_analysis(frequencies, fields, results_df)
        
        # Сохранение результатов
        logger.info("Сохранение результатов...")
        save_results(results_df, frequencies, fields, s_linear, fitted_matrix)
        
        # Итоговая статистика
        logger.info("="*60)
        logger.info("АНАЛИЗ ЗАВЕРШЕН УСПЕШНО")
        logger.info("="*60)
        logger.info(f"Проанализировано спектров: {len(results_df)}")
        logger.info(f"Среднее качество подгонки R²: {results_df['r_squared'].mean():.4f}")
        logger.info(f"Средняя когерентная связь J: {results_df['J'].mean():.4f} ГГц")
        logger.info(f"Средняя диссипативная связь Γ: {results_df['Gamma'].mean():.4f} ГГц")
        logger.info(f"Результаты сохранены в папке: {config.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        logger.error("Анализ прерван")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())