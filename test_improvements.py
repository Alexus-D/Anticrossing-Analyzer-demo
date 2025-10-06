#!/usr/bin/env python3
"""
Быстрый тест новых функций без полного анализа
"""

import numpy as np
import sys
import os

# Добавляем текущую папку в path
sys.path.insert(0, '.')

import config
import anticrossing_analyzer

def test_data_loading():
    """Тест загрузки данных с новой логикой"""
    print("Тестируем загрузку данных...")
    
    # Путь к файлу данных
    data_file = config.DATA_FILES[config.ANALYSIS_TYPE]
    data_path = os.path.join(config.DATA_DIR, data_file)
    
    if not os.path.exists(data_path):
        print(f"ОШИБКА: Файл данных не найден: {data_path}")
        return False
    
    try:
        # Загрузка с новой логикой
        frequencies, fields, s_params = anticrossing_analyzer.load_data(data_path)
        
        print(f"✓ Данные загружены успешно:")
        print(f"  - Частоты: {len(frequencies)} точек ({frequencies.min():.2f} - {frequencies.max():.2f} ГГц)")
        print(f"  - Поля: {len(fields)} точек ({fields.min():.0f} - {fields.max():.0f} Э)")
        print(f"  - S-параметры: {s_params.shape}")
        
        # Проверим, что нет полей ниже порога
        min_field = fields.min()
        if min_field >= config.MIN_FIELD_THRESHOLD:
            print(f"✓ Все поля выше порога {config.MIN_FIELD_THRESHOLD} Э (минимум: {min_field:.0f} Э)")
        else:
            print(f"✗ Найдены поля ниже порога: {min_field:.0f} Э")
        
        return True
        
    except Exception as e:
        print(f"ОШИБКА загрузки данных: {e}")
        return False

def test_magnon_frequency():
    """Тест расчета магнонной частоты"""
    print("\nТестируем расчет магнонной частоты...")
    
    test_fields = [500, 1000, 1500, 2000, 2500, 3000]
    
    print(f"Параметры: γ = {config.GYROMAGNETIC_RATIO} ГГц/Э, H_aniso = {config.ANISOTROPY_FIELD} Э")
    
    for field in test_fields:
        freq = config.calculate_magnon_frequency(field)
        print(f"  Поле {field:4d} Э -> Частота {freq:.3f} ГГц")
    
    # Проверим, что частота растет линейно
    freq1 = config.calculate_magnon_frequency(1000)
    freq2 = config.calculate_magnon_frequency(2000)
    expected_diff = config.GYROMAGNETIC_RATIO * 1000
    actual_diff = freq2 - freq1
    
    if abs(actual_diff - expected_diff) < 1e-6:
        print("✓ Линейная зависимость подтверждена")
    else:
        print(f"✗ Ошибка линейности: ожидали {expected_diff:.6f}, получили {actual_diff:.6f}")

def test_single_spectrum_fit():
    """Тест подгонки одного спектра"""
    print("\nТестируем подгонку одного спектра...")
    
    # Создаем синтетические данные
    frequencies = np.linspace(3.0, 4.0, 100)
    field_value = 2000.0
    
    # Параметры для синтетических данных
    test_params = config.INITIAL_PARAMS_SINGLE.copy()
    test_params['wm'] = config.calculate_magnon_frequency(field_value)
    
    # Генерируем теоретический спектр
    theory_complex = config.theoretical_model(frequencies, field_value, test_params, config.ANALYSIS_TYPE)
    synthetic_data = np.abs(theory_complex)
    
    # Добавляем немного шума
    noise_level = 0.01
    synthetic_data += np.random.normal(0, noise_level, synthetic_data.shape)
    
    try:
        # Пробуем подогнать
        fitted_params, fitted_spectrum, r_squared = anticrossing_analyzer.fit_spectrum(
            frequencies, synthetic_data, field_value
        )
        
        print(f"✓ Подгонка выполнена успешно:")
        print(f"  - R² = {r_squared:.4f}")
        print(f"  - Подогнанная частота магнонов: {fitted_params['wm']:.4f} ГГц")
        print(f"  - Исходная частота магнонов: {test_params['wm']:.4f} ГГц")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка подгонки: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ СИСТЕМЫ АНАЛИЗА")
    print("="*60)
    
    all_tests_passed = True
    
    # Тест 1: Загрузка данных
    if not test_data_loading():
        all_tests_passed = False
    
    # Тест 2: Расчет магнонной частоты
    test_magnon_frequency()
    
    # Тест 3: Подгонка спектра
    if not test_single_spectrum_fit():
        all_tests_passed = False
    
    print("\n" + "="*60)
    if all_tests_passed:
        print("✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("Система готова для полного анализа данных.")
    else:
        print("✗ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        print("Требуется дополнительная отладка.")
    print("="*60)
    
    return 0 if all_tests_passed else 1

if __name__ == "__main__":
    sys.exit(main())