#!/usr/bin/env python3
"""
Тестовый скрипт для проверки новых функций конфигурации
"""

import config

def test_magnon_frequency():
    """Тест функции расчета магнонной частоты"""
    print("Тест функции calculate_magnon_frequency:")
    print(f"Параметры: γ = {config.GYROMAGNETIC_RATIO} ГГц/Э, H_aniso = {config.ANISOTROPY_FIELD} Э")
    print("-" * 50)
    
    fields = [0, 500, 1000, 1500, 2000, 2500, 3000]
    for field in fields:
        freq = config.calculate_magnon_frequency(field)
        print(f"Поле {field:4d} Э -> Частота {freq:.3f} ГГц")

def test_config_params():
    """Тест новых параметров конфигурации"""
    print("\nНовые параметры конфигурации:")
    print("-" * 30)
    print(f"IGNORE_LAST_ROW: {config.IGNORE_LAST_ROW}")
    print(f"MIN_FIELD_THRESHOLD: {config.MIN_FIELD_THRESHOLD} Э")
    print(f"GYROMAGNETIC_RATIO: {config.GYROMAGNETIC_RATIO} ГГц/Э")
    print(f"ANISOTROPY_FIELD: {config.ANISOTROPY_FIELD} Э")

if __name__ == "__main__":
    test_config_params()
    test_magnon_frequency()
    print("\nВсе тесты выполнены успешно!")