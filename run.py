"""
Главный скрипт запуска пайплайна.
Последовательно запускает: Baseline -> Vanilla -> PINN.
"""
import subprocess
import sys
import os
import time


def run_script(script_path, description):
    """Запускает скрипт"""
    if not os.path.exists(script_path):
        print(f"ПРОПУСК: Файл '{script_path}' не найден.")
        return

    print(f"\nЭТАП: {description} ({script_path})")

    start_time = time.time()
    result = subprocess.run([sys.executable, script_path])
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ОШИБКА при выполнении {script_path}. Пайплайн остановлен.")
        sys.exit(1)
    else:
        print(f"ЭТАП ЗАВЕРШЕН за {elapsed:.1f} сек.")


if __name__ == "__main__":


    # 1. Baseline и EDA
    run_script("src/evaluate_all.py", "Baseline и EDA")

    # 2. Vanilla
    run_script("ML_Models/Vanilla.py", "Обучение Vanilla")

    # 3. PINN
    run_script("ML_Models/PINN_AI.py", "Обучение PINN")

    print("\nПАЙПЛАЙН ЗАВЕРШЕН!")