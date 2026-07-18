import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Análisis de Armónicos", page_icon="🌊", layout="wide")

st.title("🌊 Análisis de Armónicos — Fase A")
st.caption(
    "Composición de una señal a partir de sus armónicos individuales, con espectro de "
    "frecuencia (FFT) y cálculo de distorsión armónica total (THD)."
)

# ------------------------------------------------------------------
# Sidebar — parámetros generales
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Parámetros del sistema")

f_base = st.sidebar.slider("Frecuencia del sistema (Hz)", 10, 100, 50, step=1)
t_max = st.sidebar.slider("Duración de la señal (s)", 0.01, 1.0, 0.1, step=0.01)
fs = st.sidebar.select_slider("Frecuencia de muestreo (Hz)", options=[2000, 5000, 10000, 20000, 50000], value=10000)
num_armonicos = st.sidebar.slider("Cantidad de armónicos", 1, 20, 3)

st.sidebar.markdown("---")
preset = st.sidebar.selectbox(
    "Preset rápido",
    ["Personalizado", "Solo fundamental", "Rectificador (impares, decreciente)", "Distorsión típica motor VFD"],
)

t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

# ------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------
def valores_preset(n_arm):
    if preset == "Solo fundamental":
        mods = [1.0] + [0.0] * (n_arm - 1)
        fases = [0.0] * n_arm
    elif preset == "Rectificador (impares, decreciente)":
        mods = [1.0 if (i + 1) % 2 == 1 else 0.0 for i in range(n_arm)]
        mods = [m / (i + 1) if m else 0.0 for i, m in enumerate(mods)]
        fases = [0.0] * n_arm
    elif preset == "Distorsión típica motor VFD":
        base = {1: 1.0, 5: 0.25, 7: 0.15, 11: 0.08, 13: 0.06}
        mods = [base.get(i + 1, 0.0) for i in range(n_arm)]
        fases = [0.0] * n_arm
    else:
        mods = None
        fases = None
    return mods, fases

mods_preset, fases_preset = valores_preset(num_armonicos)

# ------------------------------------------------------------------
# Configuración de armónicos
# ------------------------------------------------------------------
st.subheader("Configuración de armónicos")

modulos, fases = [], []
n_cols = 4
cols = st.columns(n_cols)
for n in range(1, num_armonicos + 1):
    col = cols[(n - 1) % n_cols]
    with col:
        st.markdown(f"**Armónico {n}** ({n * f_base} Hz)")
        default_mod = mods_preset[n - 1] if mods_preset else (1.0 if n == 1 else 0.0)
        default_fase = fases_preset[n - 1] if fases_preset else 0.0
        modulo = st.number_input("Módulo", value=float(default_mod), step=0.05, key=f"mod_{n}", min_value=0.0)
        fase = st.number_input("Fase (°)", value=float(default_fase), step=5.0, key=f"fase_{n}")
    modulos.append(modulo)
    fases.append(np.deg2rad(fase))

# ------------------------------------------------------------------
# Construcción de señales
# ------------------------------------------------------------------
frecuencias = [n * f_base for n in range(1, num_armonicos + 1)]
armonicos_individuales = [
    A * np.sin(2 * np.pi * f_n * t + phi)
    for A, f_n, phi in zip(modulos, frecuencias, fases)
]
senal_total = np.sum(armonicos_individuales, axis=0) if armonicos_individuales else np.zeros_like(t)

# ------------------------------------------------------------------
# Métricas: RMS, THD
# ------------------------------------------------------------------
def rms(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) else 0.0

rms_total = rms(senal_total)
rms_fundamental = modulos[0] / np.sqrt(2) if num_armonicos >= 1 else 0.0
rms_armonicos_sup = np.sqrt(sum((m / np.sqrt(2)) ** 2 for m in modulos[1:])) if num_armonicos > 1 else 0.0
thd_pct = (rms_armonicos_sup / rms_fundamental * 100) if rms_fundamental > 1e-9 else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric("RMS señal total", f"{rms_total:.3f}")
m2.metric("RMS fundamental", f"{rms_fundamental:.3f}")
m3.metric("THD", f"{thd_pct:.2f} %",
          help="Distorsión armónica total: relación entre el RMS de los armónicos superiores y el RMS del fundamental.")
m4.metric("Valor pico", f"{np.max(np.abs(senal_total)) if len(senal_total) else 0:.3f}")

st.markdown("")

# ------------------------------------------------------------------
# Gráfico temporal (Plotly interactivo)
# ------------------------------------------------------------------
st.subheader("📈 Señal en el tiempo")

fig1 = go.Figure()
colores = ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"] * 4
for i, armonico in enumerate(armonicos_individuales):
    fig1.add_trace(go.Scatter(
        x=t, y=armonico, mode="lines",
        line=dict(color=colores[i % len(colores)], dash="dot", width=1.3),
        name=f"Armónico {i+1} ({frecuencias[i]} Hz)",
        opacity=0.75,
    ))
fig1.add_trace(go.Scatter(
    x=t, y=senal_total, mode="lines",
    line=dict(color="white", width=2.5),
    name="Señal total",
))
fig1.update_layout(
    xaxis_title="Tiempo (s)", yaxis_title="Amplitud",
    height=440, legend=dict(font=dict(size=9)),
    margin=dict(l=20, r=20, t=20, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig1, use_container_width=True, config={"displaylogo": False})

# ------------------------------------------------------------------
# Espectro FFT
# ------------------------------------------------------------------
st.subheader("🔍 Espectro de frecuencia (FFT)")

N = len(senal_total)
if N > 0:
    fft_vals = np.fft.fft(senal_total)
    fft_freqs = np.fft.fftfreq(N, d=1 / fs)
    fft_mags = 2.0 / N * np.abs(fft_vals[: N // 2])
    fft_freqs = fft_freqs[: N // 2]

    f_max = f_base * (num_armonicos + 1)
    mask = fft_freqs <= f_max

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=fft_freqs[mask], y=fft_mags[mask],
        marker_color="#22a884",
        hovertemplate="Frecuencia: %{x:.1f} Hz<br>Amplitud: %{y:.3f}<extra></extra>",
    ))
    fig2.update_layout(
        xaxis_title="Frecuencia (Hz)", yaxis_title="Amplitud",
        height=360, margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})
else:
    st.info("Ajustá los parámetros para generar la señal.")

# ------------------------------------------------------------------
# Tabla resumen + exportación
# ------------------------------------------------------------------
st.subheader("🧮 Resumen de armónicos")

df = pd.DataFrame({
    "Armónico": list(range(1, num_armonicos + 1)),
    "Frecuencia (Hz)": frecuencias,
    "Módulo": modulos,
    "Fase (°)": [np.rad2deg(f) for f in fases],
    "% del fundamental": [
        (m / modulos[0] * 100) if modulos[0] > 1e-9 else 0.0 for m in modulos
    ],
})
st.dataframe(df, use_container_width=True, hide_index=True)

col_dl1, col_dl2 = st.columns(2)
col_dl1.download_button(
    "⬇ Descargar configuración de armónicos (CSV)",
    df.to_csv(index=False).encode("utf-8"),
    file_name="armonicos_config.csv", mime="text/csv",
)

df_senal = pd.DataFrame({"tiempo_s": t, "amplitud": senal_total})
col_dl2.download_button(
    "⬇ Descargar señal temporal (CSV)",
    df_senal.to_csv(index=False).encode("utf-8"),
    file_name="senal_total.csv", mime="text/csv",
)

with st.expander("ℹ️ ¿Cómo se calcula el THD?"):
    st.latex(r"""
    THD = \frac{\sqrt{\sum_{n=2}^{N} V_n^2}}{V_1} \times 100\%
    """)
    st.markdown(
        "Donde $V_1$ es el valor RMS del armónico fundamental y $V_n$ el de cada "
        "armónico superior. Un THD alto indica una señal muy distorsionada respecto "
        "de una onda senoidal pura."
    )
