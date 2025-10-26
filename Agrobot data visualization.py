import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LOG_FILE = "agrobot_log.csv"
REPORT_FILE = "agrobot_report.pdf"
plt.style.use("seaborn-v0_8-darkgrid")


def main():
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print(f"Файл {LOG_FILE} не найден. Сначала запусти симуляцию и включи логирование (клавиша L).")
        return

    print(f"Загружено {len(df)} строк из {LOG_FILE}")
    print("Генерирую PDF отчёт...")

    with PdfPages(REPORT_FILE) as pdf:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df["x"], df["y"], color="#0077BB", linewidth=2.2, label="Путь робота")
        ax.scatter(df["x"].iloc[0], df["y"].iloc[0], color="#00AA55", s=80, label="Старт", zorder=5)
        ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color="#DD4444", s=80, label="Финиш", zorder=5)
        ax.set_xlabel("Координата X (м)", fontsize=11)
        ax.set_ylabel("Координата Y (м)", fontsize=11)
        ax.set_title("Траектория движения робота", fontsize=13, weight="bold")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df["t"], df["z"], color="#1f77b4", linewidth=2.0, label="Высота Z (м)")
        ax1.set_xlabel("Время (с)", fontsize=11)
        ax1.set_ylabel("Высота (м)", color="#1f77b4", fontsize=11)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(df["t"], df["moisture"], color="#2ca02c", linewidth=2.0, linestyle="--", label="Влажность почвы")
        ax2.set_ylabel("Влажность", color="#2ca02c", fontsize=11)
        ax2.tick_params(axis="y", labelcolor="#2ca02c")

        ax1.set_title("Изменения высоты и влажности почвы", fontsize=13, weight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["t"], np.degrees(df["yaw"]), label="Yaw (модель)", color="#ff7f0e", linewidth=2)
        ax.plot(df["t"], np.degrees(df["imu_yaw"]), label="Yaw (IMU)", color="#1f77b4", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Угол поворота (°)")
        ax.set_title("Изменение ориентации робота (Yaw)", fontsize=13, weight="bold")
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["t"], df["min_lidar"], color="#9467bd", linewidth=2.2)
        ax.fill_between(df["t"], df["min_lidar"], color="#c9a3e6", alpha=0.3)
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Расстояние (м)")
        ax.set_title("Минимальная дистанция до препятствий (LIDAR)", fontsize=13, weight="bold")
        pdf.savefig(fig)
        plt.close(fig)

        if "odo" in df.columns:
            dt = np.gradient(df["t"])
            dx = np.gradient(df["x"])
            dy = np.gradient(df["y"])
            vel = np.sqrt(dx**2 + dy**2) / dt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["t"], vel, color="#d62728", linewidth=2, label="Скорость (расчётная)")
            ax.plot(df["t"], df["odo"], color="#17becf", linewidth=1.8, linestyle="--", label="Одометрия")
            ax.set_xlabel("Время (с)")
            ax.set_ylabel("Скорость / путь (м)")
            ax.set_title("Динамика движения робота", fontsize=13, weight="bold")
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis("off")
        summary = [
            f"Всего записей: {len(df)}",
            f"Время моделирования: {df['t'].iloc[-1]:.1f} с",
            f"Пройденный путь (по одометрии): {df['odo'].iloc[-1]:.2f} м",
            f"Средняя влажность: {df['moisture'].mean():.3f}",
            f"Средняя высота: {df['z'].mean():.2f} м",
            f"Мин. дистанция до препятствий: {df['min_lidar'].min():.2f} м"
        ]
        text = "\n".join(summary)
        ax.text(0.1, 0.7, "Сводка данных симуляции", fontsize=16, weight="bold")
        ax.text(0.1, 0.5, text, fontsize=12, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Отчёт успешно сохранён: {REPORT_FILE}")


if __name__ == "__main__":
    main()
