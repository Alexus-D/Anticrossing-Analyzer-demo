#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работоспособности Anticrossing Analyzer
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def test_installation():
    """Проверка установки всех необходимых библиотек"""
    print("Проверка установленных библиотек...")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy не установлен")
        return False
    
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
    except ImportError:
        print("✗ SciPy не установлен")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib не установлен")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas не установлен")
        return False
    
    return True

def test_data_files():
    """Проверка наличия файлов данных"""
    print("\nПроверка файлов данных...")
    
    data_files = [
        'data/CoherentCoupling_S21.txt',
        'data/CoherentCoupling_S12.txt'
    ]
    
    all_exist = True
    for file_path in data_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # в KB
            print(f"✓ {file_path} ({size:.1f} KB)")
        else:
            print(f"✗ {file_path} не найден")
            all_exist = False
    
    return all_exist

def test_config():
    """Проверка загрузки конфигурации"""
    print("\nПроверка конфигурации...")
    
    try:
        import config
        print("✓ Файл config.py загружен успешно")
        
        # Проверка основных параметров
        required_attrs = [
            'DATA_DIR', 'RESULTS_DIR', 'DATA_FILES',
            'ANALYSIS_TYPE', 'INITIAL_PARAMS_SINGLE'
        ]
        
        for attr in required_attrs:
            if hasattr(config, attr):
                print(f"✓ Параметр {attr} найден")
            else:
                print(f"✗ Параметр {attr} отсутствует")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Ошибка загрузки config.py: {e}")
        return False

def test_results_directory():
    """Проверка директории результатов"""
    print("\nПроверка директории результатов...")
    
    results_dir = 'results'
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"✓ Директория {results_dir} существует ({len(files)} файлов)")
        
        # Показать несколько последних файлов
        if files:
            print("Последние файлы:")
            for file in sorted(files)[-5:]:
                print(f"  - {file}")
        
        return True
    else:
        print(f"✗ Директория {results_dir} не найдена")
        return False

def run_quick_test():
    """Быстрый тест основной функциональности"""
    print("\nЗапуск быстрого теста...")
    
    try:
        # Импорт основного модуля
        import anticrossing_analyzer as aa
        print("✓ Модуль anticrossing_analyzer импортирован")
        
        # Проверка основных функций
        functions = ['load_data', 'fit_spectrum', 'main']
        for func_name in functions:
            if hasattr(aa, func_name):
                print(f"✓ Функция {func_name} найдена")
            else:
                print(f"✗ Функция {func_name} отсутствует")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Ошибка импорта: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("="*60)
    print("ТЕСТ СИСТЕМЫ ANTICROSSING ANALYZER")
    print("="*60)
    
    tests = [
        ("Библиотеки", test_installation),
        ("Файлы данных", test_data_files),
        ("Конфигурация", test_config),
        ("Директория результатов", test_results_directory),
        ("Функциональность", run_quick_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"❌ Тест '{test_name}' не пройден!")
    
    print("\n" + "="*60)
    print(f"РЕЗУЛЬТАТЫ: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("🎉 Все тесты пройдены успешно!")
        print("Система готова к работе.")
        return 0
    else:
        print("⚠️  Некоторые тесты не пройдены.")
        print("Проверьте установку и конфигурацию.")
        return 1

if __name__ == "__main__":
    sys.exit(main())