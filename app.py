"""Streamlit dashboard for comparing batches of Airbnb price predictions.

The application reads ready-made CSV files and does not call S3, FastAPI,
or the training code. A CSV needs a ``predicted_price`` column.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent / "data"
PREDICTION_COLUMN = "predicted_price"
BASE_COLOR = "#648F80"
NEW_COLOR = "#BA977A"


def prepare_predictions(data: pd.DataFrame) -> tuple[pd.Series, int]:
    """Keep valid, nonnegative numeric predictions and count discarded rows."""
    if PREDICTION_COLUMN not in data.columns:
        raise ValueError(f"В CSV нет столбца '{PREDICTION_COLUMN}'.")

    values = pd.to_numeric(data[PREDICTION_COLUMN], errors="coerce")
    valid_mask = np.isfinite(values) & (values >= 0)
    cleaned = values.loc[valid_mask].astype(float).reset_index(drop=True)
    return cleaned, len(data) - len(cleaned)


def describe_predictions(values: pd.Series) -> dict[str, float | int]:
    """Summary statistics for one prediction batch."""
    return {
        "Объектов": len(values),
        "Среднее": float(values.mean()),
        "Медиана": float(values.median()),
        "Ст. отклонение": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "Минимум": float(values.min()),
        "P10": float(values.quantile(0.10)),
        "P90": float(values.quantile(0.90)),
        "Максимум": float(values.max()),
    }


def percent_difference(old: float, new: float) -> str:
    if old == 0:
        return "—" if new == 0 else "не определяется"
    return f"{(new / old - 1) * 100:+.1f}%"


def histogram_figure(samples: dict[str, pd.Series]) -> go.Figure:
    """Use shared bins so differently sized samples remain comparable."""
    all_values = np.concatenate([series.to_numpy() for series in samples.values()])
    lower, upper = float(all_values.min()), float(all_values.max())
    if upper == lower:
        lower = max(0.0, lower - 1.0)
        upper += 1.0
    bin_width = (upper - lower) / 35

    figure = go.Figure()
    for name, values in samples.items():
        figure.add_trace(
            go.Histogram(
                x=values,
                name=name,
                histnorm="percent",
                xbins=dict(start=lower, end=upper + bin_width, size=bin_width),
                opacity=0.68,
                marker_color=BASE_COLOR if name == "Базовая" else NEW_COLOR,
                hovertemplate="Цена: %{x:.1f}<br>Доля: %{y:.1f}%<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        barmode="overlay",
        xaxis_title="Предсказанная цена",
        yaxis_title="Доля объектов в интервале, %",
        legend_title_text="Выборка",
        margin=dict(l=10, r=10, t=20, b=10),
        height=390,
    )
    return figure


def boxplot_figure(samples: dict[str, pd.Series]) -> go.Figure:
    figure = go.Figure()
    for name, values in samples.items():
        figure.add_trace(
            go.Box(
                y=values,
                name=name,
                boxpoints="outliers",
                marker_color=BASE_COLOR if name == "Базовая" else NEW_COLOR,
                line_color=BASE_COLOR if name == "Базовая" else NEW_COLOR,
            )
        )
    figure.update_layout(
        yaxis_title="Предсказанная цена",
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=390,
    )
    return figure


def read_csv(file_or_path) -> pd.DataFrame:
    return pd.read_csv(file_or_path, encoding="utf-8-sig")


def main() -> None:
    st.set_page_config(page_title="Airbnb · анализ предсказаний", page_icon="🏠", layout="wide")
    st.title("Анализ предсказаний Airbnb")
    st.write(
        "Сравнение распределений предсказанной стоимости жилья. "
        "Приложение работает с готовыми CSV и не требует модели, Docker или API."
    )

    with st.sidebar:
        st.header("Данные")
        base_upload = st.file_uploader("Базовая выборка", type="csv", key="baseline")
        new_upload = st.file_uploader("Новая выборка", type="csv", key="new")
        demo_enabled = st.checkbox("Показать демонстрационные данные", value=True)
        st.caption("В каждом CSV необходим столбец `predicted_price` с числовыми прогнозами.")

    has_upload = base_upload is not None or new_upload is not None
    is_demo = not has_upload and demo_enabled
    if is_demo:
        st.info(
            "Сейчас показаны синтетические данные для демонстрации интерфейса. "
            "Это не прогнозы обученной модели. Для анализа своих результатов загрузите CSV слева."
        )
        sources = {
            "Базовая": DATA_DIR / "demo_baseline.csv",
            "Новая": DATA_DIR / "demo_new.csv",
        }
    elif has_upload:
        sources = {name: file for name, file in (("Базовая", base_upload), ("Новая", new_upload)) if file is not None}
    else:
        st.info("Загрузите один или два CSV слева либо включите демонстрационные данные.")
        st.stop()

    samples: dict[str, pd.Series] = {}
    for name, source in sources.items():
        try:
            original = read_csv(source)
            values, skipped = prepare_predictions(original)
        except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as error:
            st.error(f"{name}: не удалось прочитать данные. {error}")
            st.stop()
        if values.empty:
            st.error(f"{name}: в столбце '{PREDICTION_COLUMN}' нет корректных числовых значений.")
            st.stop()
        samples[name] = values
        if skipped:
            st.warning(f"{name}: исключено {skipped} строк с пустыми, нечисловыми или отрицательными значениями.")

    st.subheader("Основные показатели")
    cols = st.columns(4)
    base = samples.get("Базовая")
    new = samples.get("Новая")
    current = base if base is not None else new
    assert current is not None
    stats = describe_predictions(current)
    cols[0].metric("Объектов", f"{stats['Объектов']:,}".replace(",", " "))
    cols[1].metric("Медиана", f"{stats['Медиана']:,.2f}")
    cols[2].metric("Среднее", f"{stats['Среднее']:,.2f}")
    cols[3].metric("90-й перцентиль", f"{stats['P90']:,.2f}")
    st.caption("В карточках показана базовая выборка, а если она не загружена — новая.")

    st.subheader("Статистика по выборкам")
    summary = pd.DataFrame({name: describe_predictions(values) for name, values in samples.items()}).T
    summary["Объектов"] = summary["Объектов"].astype(int)
    st.dataframe(summary.round(2), use_container_width=True)

    st.subheader("Распределение предсказанных цен")
    st.plotly_chart(histogram_figure(samples), use_container_width=True)
    st.caption("Показана доля объектов в каждом интервале, поэтому выборки разного размера можно сравнивать.")

    st.subheader("Разброс и выбросы")
    st.plotly_chart(boxplot_figure(samples), use_container_width=True)

    if base is not None and new is not None:
        st.subheader("Сравнение базовой и новой выборок")
        base_stats = describe_predictions(base)
        new_stats = describe_predictions(new)
        compare = pd.DataFrame(
            [
                {
                    "Показатель": name,
                    "Базовая": base_stats[name],
                    "Новая": new_stats[name],
                    "Изменение": percent_difference(float(base_stats[name]), float(new_stats[name])),
                }
                for name in ("Среднее", "Медиана", "P90")
            ]
        )
        st.dataframe(compare.round(2), hide_index=True, use_container_width=True)
        st.write(
            "Разница медиан: "
            f"{percent_difference(float(base_stats['Медиана']), float(new_stats['Медиана']))}. "
            "Это показывает изменение распределения прогнозов, но не качество модели. "
            "Для оценки MAE или RMSE потребовались бы реальные цены."
        )
    else:
        st.info("Чтобы сравнить две выборки, загрузите также второй CSV.")

    with st.expander("Посмотреть первые строки"):
        for name, values in samples.items():
            st.write(name)
            st.dataframe(values.rename(PREDICTION_COLUMN).head(10).to_frame(), hide_index=True)


if __name__ == "__main__":
    main()
