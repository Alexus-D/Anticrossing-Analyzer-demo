#!/usr/bin/env python3
"""
Тестовый скрипт для проверки улучшений в конфигурации
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import numpy as np

def test_new_fmr_calculation():
    """Тест новой системы расчета частоты ФМР через калибровочную точку"""
    print("="*60)
    print("ТЕСТ НОВОЙ СИСТЕМЫ РАСЧЕТА ЧАСТОТЫ ФМР")
    print("="*60)
    
    print(f"Калибровочная точка: {config.FMR_CALIBRATION_FIELD} Э -> {config.FMR_CALIBRATION_FREQUENCY} ГГц")
    print(f"Гиромагнитное отношение: {config.GYROMAGNETIC_RATIO} ГГц/Э")
    print("-" * 40)
    
    # Тест различных полей
    test_fields = [2800, 2900, 3000, 3100, 3200]
    
    for field in test_fields:
        freq = config.calculate_magnon_frequency(field)
        print(f"Поле {field:4d} Э -> Частота {freq:.3f} ГГц")
    
    # Проверим, что калибровочная точка дает правильный результат
    calib_freq = config.calculate_magnon_frequency(config.FMR_CALIBRATION_FIELD)
    if abs(calib_freq - config.FMR_CALIBRATION_FREQUENCY) < 1e-10:
        print("✓ Калибровочная точка работает корректно")
    else:
        print(f"✗ Ошибка калибровки: ожидалось {config.FMR_CALIBRATION_FREQUENCY}, получено {calib_freq}")

def test_new_parameters():
    """Тест новых параметров конфигурации"""
    print("\n" + "="*60)
    print("ТЕСТ НОВЫХ ПАРАМЕТРОВ КОНФИГУРАЦИИ")
    print("="*60)
    
    print("Параметры ширины линий:")
    print(f"  Резонатор: {config.CAVITY_LINEWIDTH} ГГц")
    print(f"  Магноны:   {config.MAGNON_LINEWIDTH} ГГц")
    
    print("\nНачальные параметры:")
    for key, value in config.INITIAL_PARAMS_SINGLE.items():
        print(f"  {key:12s}: {value:.3f}")
    
    print("\nГраницы параметров:")
    for key, bounds in config.PARAM_BOUNDS.items():
        print(f"  {key:12s}: [{bounds[0]:.3f}, {bounds[1]:.3f}]")

def test_theoretical_model():
    """Тест теоретической модели с новыми параметрами"""
    print("\n" + "="*60)
    print("ТЕСТ ТЕОРЕТИЧЕСКОЙ МОДЕЛИ")
    print("="*60)
    
    # Тестовые параметры
    frequencies = np.linspace(3.5, 3.8, 100)
    field_value = 3000
    test_params = config.INITIAL_PARAMS_SINGLE.copy()
    test_params['wm'] = config.calculate_magnon_frequency(field_value)
    
    try:
        # Вычисление S-параметров
        s_params = config.theoretical_model(frequencies, field_value, test_params, 'S21')
        
        print(f"✓ Теоретическая модель работает корректно")
        print(f"  Размер выходного массива: {s_params.shape}")
        print(f"  Тип данных: {s_params.dtype}")
        print(f"  Диапазон амплитуд: {np.abs(s_params).min():.3f} - {np.abs(s_params).max():.3f}")
        
    except Exception as e:
        print(f"✗ Ошибка в теоретической модели: {e}")

def main():
    """Главная функция тестирования"""
    test_new_fmr_calculation()
    test_new_parameters()
    test_theoretical_model()
    
    print("\n" + "="*60)
    print("✓ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
    print("="*60)

if __name__ == "__main__":
    main()