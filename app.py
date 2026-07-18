import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

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

f_base = st.sidebar.slider("Frecuencia del sistema (Hz)", 10, 100, 50, step=1, key="f_base")
num_ciclos = st.sidebar.slider(
    "Duración — ciclos de la fundamental", 1, 20, 5, step=1, key="num_ciclos",
    help="Se expresa en ciclos enteros (no en segundos) para que la señal analizada "
         "sea siempre un número exacto de períodos. Esto evita la fuga espectral "
         "(spectral leakage) en la FFT: con una duración arbitraria en segundos, si no "
         "coincide con un múltiplo exacto del período, los picos del espectro salen "
         "'corridos' en vez de nítidos.",
)
fs = st.sidebar.select_slider("Frecuencia de muestreo (Hz)", options=[2000, 5000, 10000, 20000, 50000], value=10000, key="fs")
num_armonicos = st.sidebar.slider("Cantidad de armónicos", 1, 20, 3, key="num_arm")

t_max = num_ciclos / f_base

st.sidebar.markdown("---")
preset = st.sidebar.selectbox(
    "Preset rápido",
    ["Personalizado", "Solo fundamental", "Rectificador (impares, decreciente)", "Distorsión típica motor VFD"],
    key="preset_select",
)


def _aplicar_preset():
    p = st.session_state["preset_select"]
    n_arm = st.session_state["num_arm"]
    if p == "Solo fundamental":
        mods = [1.0] + [0.0] * (n_arm - 1)
        fases = [0.0] * n_arm
    elif p == "Rectificador (impares, decreciente)":
        mods = [(1.0 / (i + 1)) if (i + 1) % 2 == 1 else 0.0 for i in range(n_arm)]
        fases = [0.0] * n_arm
    elif p == "Distorsión típica motor VFD":
        base = {1: 1.0, 5: 0.25, 7: 0.15, 11: 0.08, 13: 0.06}
        mods = [base.get(i + 1, 0.0) for i in range(n_arm)]
        fases = [0.0] * n_arm
    else:
        return  # "Personalizado": no toca nada, deja los valores actuales
    for i in range(n_arm):
        st.session_state[f"mod_{i+1}"] = mods[i]
        st.session_state[f"fase_{i+1}"] = fases[i]


st.sidebar.button("✅ Aplicar preset", on_click=_aplicar_preset, use_container_width=True)

# Nyquist: la frecuencia del armónico más alto no puede superar fs/2
f_max_pedido = f_base * num_armonicos
if f_max_pedido >= fs / 2:
    st.sidebar.warning(
        f"⚠️ El armónico más alto pedido ({f_max_pedido} Hz) supera la frecuencia de "
        f"Nyquist para esta frecuencia de muestreo ({fs/2:.0f} Hz). Subí `fs` o bajá la "
        "cantidad de armónicos, o el espectro mostrado va a tener aliasing."
    )

t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

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
        default_mod = 1.0 if n == 1 else 0.0
        modulo = st.number_input("Módulo", value=default_mod, step=0.05, key=f"mod_{n}", min_value=0.0)
        fase = st.number_input("Fase (°)", value=0.0, step=5.0, key=f"fase_{n}")
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
    fft_mags[0] /= 2.0  # el bin de DC no se refleja, no lleva el factor 2 del resto
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

# ------------------------------------------------------------------
# Síntesis animada (epiciclos de Fourier)
# ------------------------------------------------------------------
# Misma técnica que la app de componentes simétricas: componente HTML
# embebido con Plotly.js corriendo 100% en el navegador, animado con
# requestAnimationFrame. Cada armónico es un fasor que gira a su propia
# velocidad (n·f_base) y se van encadenando punta-con-cola; la punta del
# último vector traza exactamente la señal compuesta en el tiempo — es
# la construcción geométrica clásica de una serie de Fourier.
st.subheader("🌀 Síntesis animada (epiciclos de Fourier)")
st.caption(
    "Cada armónico es un vector que gira a su propia velocidad. Encadenados punta con "
    "cola, la punta del último traza exactamente la señal compuesta a la derecha."
)

armonicos_json = [
    {"n": n, "mod": modulos[n - 1], "fase": float(fases[n - 1])}
    for n in range(1, num_armonicos + 1)
]
colores_json = [colores[i % len(colores)] for i in range(num_armonicos)]

EPI_PARAMS = dict(fBase=f_base, armonicos=armonicos_json, colores=colores_json)

EPI_HTML = r"""
<div style="font-family:'Source Sans Pro',sans-serif; color:#fafafa;">
  <div style="display:flex; gap:16px; flex-wrap:wrap;">
    <div id="epi" style="flex:1; min-width:320px; height:380px;"></div>
    <div id="ondaAnim" style="flex:1.3; min-width:380px; height:380px;"></div>
  </div>
  <div style="margin:8px 0; display:flex; align-items:center; gap:12px; flex-wrap:wrap;">
    <button id="epiPlay" style="background:#2b2f38; color:#fafafa; border:1px solid #555; border-radius:6px; padding:6px 16px; cursor:pointer; font-size:14px;">▶ Animar</button>
    <label style="font-size:13px; color:#bbb; display:flex; align-items:center; gap:6px;">
      Velocidad
      <input id="epiSpeed" type="range" min="0.02" max="1" step="0.02" value="0.15" style="width:100px;">
      <span id="epiSpeedLabel">0.15×</span>
    </label>
  </div>
</div>

<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
const EP = __EPI_PARAMS__;
const OMEGA_BASE = 2 * Math.PI * EP.fBase;

function cAdd(a,b){ return {re:a.re+b.re, im:a.im+b.im}; }
function cAbs(a){ return Math.hypot(a.re,a.im); }

function fasorArm(h, tSec) {
  const angRad = h.fase + OMEGA_BASE * h.n * tSec;
  return {re: h.mod*Math.cos(angRad), im: h.mod*Math.sin(angRad)};
}

// --- onda estática de referencia (2 períodos de la fundamental) ---
const T_PERIOD_MS = (1/EP.fBase) * 1000;
const T_ARRAY = [];
for (let i=0; i<600; i++) T_ARRAY.push(i * (2*T_PERIOD_MS) / 599);

function ondaEnT(tMs) {
  const t = tMs/1000;
  let y = 0;
  EP.armonicos.forEach(h => { y += h.mod * Math.sin(OMEGA_BASE*h.n*t + h.fase); });
  return y;
}
const ONDA_Y = T_ARRAY.map(ondaEnT);
const yMax = Math.max(...ONDA_Y.map(Math.abs), 1) * 1.25;

const modSum = EP.armonicos.reduce((s,h)=>s+h.mod, 0) || 1;
const limEpi = modSum * 1.15;

const BASE_LAYOUT = {
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{color:'#eee'}, margin:{l:40,r:20,t:36,b:36}, showlegend:false,
};
const AX = {zeroline:false, showgrid:true, gridcolor:'rgba(128,128,128,0.25)', color:'#bbb'};

let initialized = false;

function chainTraces(tSec) {
  let p = {re:0, im:0};
  const xs=[0], ys=[0];
  EP.armonicos.forEach(h => {
    const V = fasorArm(h, tSec);
    p = cAdd(p, V);
    xs.push(p.re); ys.push(p.im);
  });
  return {xs, ys, tip: p};
}

function render(tMs) {
  const tSec = tMs/1000;
  const {xs, ys, tip} = chainTraces(tSec);

  const epiTraces = [{
    x: xs, y: ys, mode:'lines+markers',
    line:{color:'rgba(200,200,200,0.5)', width:1.5},
    marker:{size:5, color: ['#888'].concat(EP.colores)},
  }];
  const epiLayout = Object.assign({}, BASE_LAYOUT, {
    title:{text:'Suma vectorial (epiciclos)', font:{size:13}},
    xaxis: Object.assign({}, AX, {range:[-limEpi, limEpi]}),
    yaxis: Object.assign({}, AX, {range:[-limEpi, limEpi], scaleanchor:'x', scaleratio:1}),
  });

  const ondaLayout = Object.assign({}, BASE_LAYOUT, {
    title:{text:'Señal compuesta', font:{size:13}},
    xaxis: Object.assign({}, AX, {title:'Tiempo (ms)'}),
    yaxis: Object.assign({}, AX, {range:[-yMax, yMax]}),
    shapes:[{type:'line', x0:tMs, x1:tMs, y0:-yMax, y1:yMax,
             line:{color:'rgba(255,255,255,0.4)', width:1, dash:'dash'}}],
  });

  const config = {displaylogo:false, responsive:true};

  if (!initialized) {
    Plotly.newPlot('epi', epiTraces, epiLayout, config);
    Plotly.newPlot('ondaAnim', [
      {x:T_ARRAY, y:ONDA_Y, mode:'lines', line:{color:'#eee', width:2}},
      {x:[tMs], y:[tip.im], mode:'markers', marker:{size:9, color:'#fff'}},
    ], ondaLayout, config);
    initialized = true;
  } else {
    Plotly.react('epi', epiTraces, epiLayout, config);
    Plotly.relayout('ondaAnim', {shapes: ondaLayout.shapes});
    Plotly.restyle('ondaAnim', {x:[[tMs]], y:[[tip.im]]}, [1]);
  }
}

let animando=false, lastT=null, tMs=0, velocidad=0.15*0.05;
const playBtn = document.getElementById('epiPlay');
const speedSlider = document.getElementById('epiSpeed');
const speedLabel = document.getElementById('epiSpeedLabel');
const CICLO_MS = 2 * T_PERIOD_MS;

function frame(now) {
  if (!animando) return;
  if (lastT===null) lastT = now;
  const dt = now - lastT; lastT = now;
  tMs = (tMs + dt*velocidad) % CICLO_MS;
  render(tMs);
  requestAnimationFrame(frame);
}
playBtn.addEventListener('click', () => {
  animando = !animando;
  playBtn.textContent = animando ? '⏸ Pausar' : '▶ Animar';
  if (animando) { lastT=null; requestAnimationFrame(frame); }
});
speedSlider.addEventListener('input', () => {
  const v = parseFloat(speedSlider.value);
  velocidad = v*0.05;
  speedLabel.textContent = v.toFixed(2) + '×';
});

render(0);
</script>
"""

epi_html_final = EPI_HTML.replace("__EPI_PARAMS__", json.dumps(EPI_PARAMS))
components.html(epi_html_final, height=480, scrolling=False)
