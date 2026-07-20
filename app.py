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
num_armonicos = st.sidebar.number_input(
    "Cantidad de armónicos", min_value=1, max_value=200, value=3, step=1, key="num_arm",
    help="Sin límite práctico — para números grandes la edición se hace más cómoda "
         "en la tabla de abajo que con un slider.",
)

t_max = num_ciclos / f_base

st.sidebar.markdown("---")
preset = st.sidebar.selectbox(
    "Preset rápido",
    ["Personalizado", "Solo fundamental", "Onda cuadrada (impares 1/n)", "Distorsión típica motor VFD"],
    key="preset_select",
)


def _armonicos_default(p: str, n_arm: int):
    """Devuelve (mods, fases_deg) para un preset dado, sin tocar session_state."""
    if p == "Solo fundamental":
        mods = [1.0] + [0.0] * (n_arm - 1)
    elif p == "Onda cuadrada (impares 1/n)":
        mods = [(1.0 / (i + 1)) if (i + 1) % 2 == 1 else 0.0 for i in range(n_arm)]
    elif p == "Distorsión típica motor VFD":
        base = {1: 1.0, 5: 0.25, 7: 0.15, 11: 0.08, 13: 0.06}
        mods = [base.get(i + 1, 0.0) for i in range(n_arm)]
    else:  # Personalizado / fallback
        mods = [1.0 if i == 0 else 0.0 for i in range(n_arm)]
    fases = [0.0] * n_arm
    return mods, fases


def _aplicar_preset():
    p = st.session_state["preset_select"]
    n_arm = st.session_state["num_arm"]
    if p == "Personalizado":
        return  # no toca nada, deja los valores actuales
    mods, fases_deg = _armonicos_default(p, n_arm)
    st.session_state["harm_df"] = pd.DataFrame({
        "Armónico": list(range(1, n_arm + 1)),
        "Frecuencia (Hz)": [n * st.session_state["f_base"] for n in range(1, n_arm + 1)],
        "Módulo": mods,
        "Fase (°)": fases_deg,
    })
    # Descarta cualquier edición manual pendiente en la tabla para que
    # tome los valores del preset sin que el widget la pise de vuelta
    # (mismo problema que ya resolvimos con los sliders de la app de fasores).
    st.session_state.pop("harm_editor", None)


st.sidebar.button("✅ Aplicar preset", on_click=_aplicar_preset, use_container_width=True)

# Nyquist: la frecuencia del armónico más alto CON AMPLITUD NO NULA no
# puede superar fs/2. El chequeo real se hace más abajo (tras leer la
# tabla); acá sólo se reserva el lugar en el sidebar.
nyquist_ph = st.sidebar.empty()

t = np.linspace(0, t_max, int(fs * t_max), endpoint=False)

# ------------------------------------------------------------------
# Configuración de armónicos — tabla editable
# ------------------------------------------------------------------
st.subheader("Configuración de armónicos")
st.caption("Editá Módulo y Fase directamente en la tabla. Frecuencia se recalcula sola.")

frecuencias_actuales = [n * f_base for n in range(1, num_armonicos + 1)]

if "harm_df" not in st.session_state:
    mods0, fases0 = _armonicos_default("Personalizado", num_armonicos)
    st.session_state["harm_df"] = pd.DataFrame({
        "Armónico": list(range(1, num_armonicos + 1)),
        "Frecuencia (Hz)": frecuencias_actuales,
        "Módulo": mods0,
        "Fase (°)": fases0,
    })
else:
    # Ajustar cantidad de filas si cambió num_armonicos, preservando lo ya cargado
    base_df = st.session_state["harm_df"]
    n_actual = len(base_df)
    if n_actual != num_armonicos:
        if num_armonicos > n_actual:
            extra_mods, extra_fases = _armonicos_default("Personalizado", num_armonicos - n_actual)
            extra = pd.DataFrame({
                "Armónico": list(range(n_actual + 1, num_armonicos + 1)),
                "Frecuencia (Hz)": [n * f_base for n in range(n_actual + 1, num_armonicos + 1)],
                "Módulo": [0.0] * (num_armonicos - n_actual),
                "Fase (°)": [0.0] * (num_armonicos - n_actual),
            })
            base_df = pd.concat([base_df, extra], ignore_index=True)
        else:
            base_df = base_df.iloc[:num_armonicos].reset_index(drop=True)
        base_df["Frecuencia (Hz)"] = frecuencias_actuales
        st.session_state["harm_df"] = base_df
        st.session_state.pop("harm_editor", None)
    else:
        # misma cantidad de filas: sólo refrescar frecuencia por si cambió f_base
        base_df["Frecuencia (Hz)"] = frecuencias_actuales
        st.session_state["harm_df"] = base_df

edited_df = st.data_editor(
    st.session_state["harm_df"],
    key="harm_editor",
    use_container_width=True,
    hide_index=True,
    height=min(38 * (num_armonicos + 1) + 3, 480),
    column_config={
        "Armónico": st.column_config.NumberColumn(disabled=True),
        "Frecuencia (Hz)": st.column_config.NumberColumn(disabled=True, format="%d"),
        "Módulo": st.column_config.NumberColumn(min_value=0.0, step=0.05, format="%.3f"),
        "Fase (°)": st.column_config.NumberColumn(step=5.0, format="%.1f"),
    },
)
# Red de seguridad: si el usuario borra una celda queda NaN y se
# propaga a toda la señal / THD / FFT. Se rellena con 0.
edited_df[["Módulo", "Fase (°)"]] = edited_df[["Módulo", "Fase (°)"]].fillna(0.0)
st.session_state["harm_df"] = edited_df

modulos = edited_df["Módulo"].tolist()
fases = np.deg2rad(edited_df["Fase (°)"].to_numpy()).tolist()

# ------------------------------------------------------------------
# Construcción de señales
# ------------------------------------------------------------------
frecuencias = [n * f_base for n in range(1, num_armonicos + 1)]

# La señal total se acumula en un solo array en vez de materializar los
# (hasta 200) armónicos a la vez: mucho más liviano en memoria y CPU.
senal_total = np.zeros_like(t)
for A, f_n, phi in zip(modulos, frecuencias, fases):
    if A != 0.0:
        senal_total += A * np.sin(2 * np.pi * f_n * t + phi)

# Chequeo de Nyquist sobre el armónico más alto realmente activo.
armonicos_activos = [n for n, A in zip(range(1, num_armonicos + 1), modulos) if abs(A) > 1e-9]
f_max_pedido = f_base * (max(armonicos_activos) if armonicos_activos else 1)
if f_max_pedido >= fs / 2:
    nyquist_ph.warning(
        f"⚠️ El armónico activo más alto ({f_max_pedido:.0f} Hz) supera la frecuencia de "
        f"Nyquist para esta frecuencia de muestreo ({fs/2:.0f} Hz). Subí `fs` o bajá el "
        "armónico más alto, o el espectro mostrado va a tener aliasing."
    )

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

# Sólo se trazan los armónicos con amplitud no nula, y hasta un tope,
# para no saturar el navegador ni la leyenda. La señal total siempre
# incluye a todos.
MAX_TRAZAS = 15
idx_activos = [i for i, A in enumerate(modulos) if abs(A) > 1e-9]
idx_dibujar = idx_activos[:MAX_TRAZAS]
for i in idx_dibujar:
    armonico = modulos[i] * np.sin(2 * np.pi * frecuencias[i] * t + fases[i])
    fig1.add_trace(go.Scatter(
        x=t, y=armonico, mode="lines",
        line=dict(color=colores[i % len(colores)], dash="dot", width=1.3),
        name=f"Armónico {i+1} ({frecuencias[i]} Hz)",
        opacity=0.75,
    ))
if len(idx_activos) > MAX_TRAZAS:
    st.caption(
        f"Se muestran los primeros {MAX_TRAZAS} de {len(idx_activos)} armónicos activos "
        "como trazas individuales; la señal total (blanca) los incluye a todos."
    )
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
    # Paso de muestreo REAL (t_max/N). Como N = int(fs*t_max) trunca la
    # fracción, usar 1/fs desalinea las etiquetas del eje cuando fs*t_max
    # no es entero (los picos siguen on-bin, pero mal rotulados).
    dt_real = (t[1] - t[0]) if N > 1 else 1 / fs
    fft_freqs = np.fft.fftfreq(N, d=dt_real)
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
# Resumen (% del fundamental) + exportación
# ------------------------------------------------------------------
st.subheader("🧮 Resumen — % respecto del fundamental")

df = edited_df.copy()
df["% del fundamental"] = (
    df["Módulo"] / df["Módulo"].iloc[0] * 100 if df["Módulo"].iloc[0] > 1e-9 else 0.0
)
st.dataframe(df, use_container_width=True, hide_index=True,
             height=min(38 * (num_armonicos + 1) + 3, 320))

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

if num_armonicos > 60:
    st.info(
        f"Con {num_armonicos} armónicos el diagrama de epiciclos se vuelve ilegible y "
        "pesado para el navegador — esta visualización se muestra hasta 60 armónicos. "
        "El resto de los cálculos (señal, FFT, THD) no tiene este límite."
    )
else:
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
