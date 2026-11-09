import math
import os
import json
import statistics
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
try:
    from scipy.stats import norm, t
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

from pye import arboles, conteo, distribuciones, frecuencias, inferencia, probabilidad, regresion, variables

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Solver Estadístico Unificado",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-title {font-size: 42px; font-weight: 800; margin-bottom: 0px;}
    .sub-title {font-size: 18px; color: #9CA3AF; margin-bottom: 22px;}
    .metric-card {
        background-color: rgba(120, 120, 120, 0.10);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(150, 150, 150, 0.25);
        margin-bottom: 12px;
        min-height: 135px;
    }
    .report-box {
        background-color: rgba(34, 197, 94, 0.10);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(34, 197, 94, 0.35);
        margin-top: 15px;
    }
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 22px;
        margin-top: 10px;
        margin-bottom: 15px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.15);
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .section-chip {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.20);
        color: #dbeafe;
        font-size: 0.85rem;
        padding: 6px 10px;
        border-radius: 999px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FUNCIONES AUXILIARES GENERALES
# ============================================================

def parse_num(valor):
    if isinstance(valor, (int, float, np.integer, np.floating)):
        return float(valor)
    s = str(valor).strip().replace(" ", "")
    if s == "":
        raise ValueError("Entrada vacía")
    if s.endswith("%"):
        s = s[:-1]
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        return float(s) / 100
    if s.count(".") >= 2 and "," not in s:
        s = s.replace(".", "")
        return float(s)
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    return float(s)


def phi(z):
    if SCIPY_OK:
        return norm.cdf(z)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def inv_norm(p):
    if SCIPY_OK:
        return norm.ppf(p)
    low, high = -5, 5
    for _ in range(100):
        mid = (low + high) / 2
        if phi(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def fmt_prob(p):
    if p < 0 or p > 1:
        return f"⚠️ Valor fuera de rango ({p:.6f}) — revisa los límites a y b"
    if p < 1e-6:
        return "< 0.000001 (<0.0001%)"
    return f"{p:.6f} ({p*100:.2f}%)"


def interpretar_correlacion(r):
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return "indefinida", "una de las variables no tiene variabilidad (σ = 0)"
    
    if r > 0:
        tipo = "positiva"
    elif r < 0:
        tipo = "negativa"
    else:
        tipo = "nula"

    fuerza = abs(r)
    if fuerza == 1:
        intensidad = "perfecta"
    elif fuerza >= 0.80:
        intensidad = "muy fuerte"
    elif fuerza >= 0.60:
        intensidad = "fuerte"
    elif fuerza >= 0.40:
        intensidad = "moderada"
    elif fuerza >= 0.20:
        intensidad = "débil"
    else:
        intensidad = "muy débil o casi inexistente"
    return tipo, intensidad


def p_valor_normal(z, tipo):
    if tipo == "Bilateral":
        return 2 * (1 - phi(abs(z)))
    if tipo == "Cola izquierda":
        return phi(z)
    return 1 - phi(z)


def p_valor_t(t_calc, gl, tipo):
    if SCIPY_OK:
        if tipo == "Bilateral":
            return 2 * (1 - t.cdf(abs(t_calc), gl))
        if tipo == "Cola izquierda":
            return t.cdf(t_calc, gl)
        return 1 - t.cdf(t_calc, gl)
    return p_valor_normal(t_calc, tipo)


def critico_z(alpha, tipo):
    if tipo == "Bilateral":
        return abs(inv_norm(1 - alpha / 2))
    if tipo == "Cola izquierda":
        return inv_norm(alpha)
    return abs(inv_norm(1 - alpha))


def critico_t(alpha, gl, tipo):
    if SCIPY_OK:
        if tipo == "Bilateral":
            return t.ppf(1 - alpha / 2, gl)
        if tipo == "Cola izquierda":
            return t.ppf(alpha, gl)
        return t.ppf(1 - alpha, gl)
    return critico_z(alpha, tipo)


def conclusion_hipotesis(rechaza, alpha, contexto, h1_texto):
    nivel = (1 - alpha) * 100
    if rechaza:
        return (
            f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, "
            f"se rechaza la hipótesis nula. Por tanto, existe evidencia estadística suficiente para afirmar que "
            f"{h1_texto}. En el contexto del problema, esto indica que {contexto}"
        )
    return (
        f"Con un nivel de significancia de {alpha:.2f} y un nivel de confianza aproximado de {nivel:.0f}%, "
        f"no se rechaza la hipótesis nula. Por tanto, no existe evidencia estadística suficiente para afirmar que "
        f"{h1_texto}. En el contexto del problema, esto indica que {contexto}"
    )


def show_report(texto):
    st.markdown("### Conclusión para informe")
    st.markdown(f"<div class='report-box'>{texto}</div>", unsafe_allow_html=True)
    st.download_button(
        "Descargar conclusión en TXT",
        data=texto,
        file_name="conclusion_estadistica.txt",
        mime="text/plain"
    )


@st.cache_data
def leer_csv_excel_cached(contenido: bytes, nombre: str):
    import io
    archivo = io.BytesIO(contenido)
    archivo.name = nombre
    if nombre.endswith(".csv"):
        intentos = [
            {"sep": ";", "decimal": ","},
            {"sep": ",", "decimal": "."},
            {"sep": None, "decimal": ","},
            {"sep": None, "decimal": "."}
        ]
        for opt in intentos:
            try:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=opt["sep"], decimal=opt["decimal"], engine='python' if opt["sep"] is None else None)
                if not df.empty and df.columns.size > 0:
                    return df
            except Exception:
                continue
        archivo.seek(0)
        return pd.read_csv(archivo)
    
    if nombre.endswith(".xlsx") or nombre.endswith(".xls"):
        return pd.read_excel(archivo)
    raise ValueError("Formato no soportado. Usa CSV o Excel.")


def plot_heatmap(matriz):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matriz, aspect="auto")
    ax.set_xticks(range(len(matriz.columns)))
    ax.set_yticks(range(len(matriz.columns)))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right")
    ax.set_yticklabels(matriz.columns)
    for i in range(len(matriz.columns)):
        for j in range(len(matriz.columns)):
            ax.text(j, i, f"{matriz.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de correlación de Pearson")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_section_header(title, subtitle, icon="📊", show_chip=True):
    chip_html = '<div class="section-chip">Módulo activo</div>' if show_chip else ''
    st.markdown(
        f"""
        {chip_html}
        <div style="display: flex; align-items: center; gap: 18px; margin-bottom: 12px;">
            <div style="font-size: 2rem; background: rgba(56, 189, 248, 0.10); padding: 10px 14px; border-radius: 16px;">{icon}</div>
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">{title}</h1>
                <p style="margin: 0; color: #64748b;">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_topic_selector(options, key):
    st.markdown("### Tema")
    return st.selectbox("Selecciona el tema a trabajar", options, key=key, label_visibility="collapsed")


def input_num(label, default, key=None, minimo=None, maximo=None):
    txt = st.text_input(label, str(default), key=key)
    try:
        v = parse_num(txt)
        if minimo is not None and v < minimo:
            st.error(f"'{label}' debe ser ≥ {minimo}.")
            return parse_num(default)
        if maximo is not None and v > maximo:
            st.error(f"'{label}' debe ser ≤ {maximo}.")
            return parse_num(default)
        return v
    except ValueError:
        st.error(f"'{label}': valor no numérico. Se usa {default}.")
        return parse_num(default)

def render_three_inputs(labels_defaults):
    cols = st.columns(3)
    values = []
    for col, (label, default) in zip(cols, labels_defaults):
        with col:
            values.append(input_num(label, default))
    return values


def render_four_inputs(labels_defaults):
    cols = st.columns(4)
    values = []
    for col, (label, default) in zip(cols, labels_defaults):
        with col:
            values.append(input_num(label, default))
    return values


def render_result_box(title, value):
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.16), rgba(56, 189, 248, 0.04));
                    border: 1px solid rgba(56, 189, 248, 0.25); padding: 18px; border-radius: 16px; margin-top: 12px; margin-bottom: 16px;">
            <div style="font-size: 0.95rem; color: #93c5fd; font-weight: 700; margin-bottom: 6px;">{title}</div>
            <div style="font-size: 1.45rem; font-weight: 800;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FUNCIONES ACADÉMICAS ADICIONALES
# ============================================================

def calcular_moda(datos):
    conteos = Counter(datos)
    max_frecuencia = max(conteos.values())
    modas = [k for k, v in conteos.items() if v == max_frecuencia]
    return modas


def calcular_percentil(datos_ordenados, p):
    n = len(datos_ordenados)
    if n == 0:
        return 0.0
    if n == 1:
        return datos_ordenados[0]
    idx = (p / 100) * (n - 1)
    idx_inf = int(math.floor(idx))
    idx_sup = int(math.ceil(idx))
    peso = idx - idx_inf
    if idx_inf == idx_sup:
        return datos_ordenados[idx_inf]
    return datos_ordenados[idx_inf] * (1 - peso) + datos_ordenados[idx_sup] * peso


def permutacion_lineal_sin_rep(n):
    return math.factorial(n)


def permutacion_con_rep_identicos(n, frecuencias):
    denominador = 1
    for f in frecuencias:
        denominador *= math.factorial(f)
    return math.factorial(n) // denominador


def variacion_sin_rep(n, r):
    return math.factorial(n) // math.factorial(n - r) if r <= n else 0


def variacion_con_rep(n, r):
    return n ** r


def combinacion_sin_rep(n, r):
    return math.comb(n, r) if r <= n else 0


def combinacion_con_rep(n, r):
    return math.comb(n + r - 1, r)


def funcion_error_aprox(x):
    signo = 1 if x >= 0 else -1
    x = abs(x)
    t_aux = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t_aux - 1.453152027) * t_aux) + 1.421413741) * t_aux - 0.284496736) * t_aux + 0.254829592) * t_aux * math.exp(-x * x)
    return signo * y


def cdf_normal_estandar(z):
    return phi(z)


def pdf_normal(x, mu, sigma):
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def probabilidad_binomial(k, n, p):
    if n > 170:
        return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log(1 - p))
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def probabilidad_poisson(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def get_z_value(confianza):
    alfa = 1.0 - confianza
    p = 1 - alfa / 2
    t_aux = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t_aux - ((c2*t_aux + c1)*t_aux + c0) / (((d3*t_aux + d2)*t_aux + d1)*t_aux + 1.0)
    return z


def get_t_value(confianza, df):
    z = get_z_value(confianza)
    if df >= 30:
        return z
    return z + (z**3 + z) / (4.0 * df)


def registrar_calculo(modulo, inputs_dict):
    try:
        if "historial" not in st.session_state:
            st.session_state["historial"] = []
        log_entry = {
            "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modulo": modulo,
            "inputs": inputs_dict
        }
        st.session_state["historial"].append(log_entry)
        return True
    except Exception:
        return False

# ============================================================
# NAVEGACIÓN
# ============================================================

MENU = [
    "Inicio",
    "Indicadores Estadísticos",
    "Tabla de Frecuencias",
    "Probabilidad Básica",
    "Reglas de Conteo",
    "Árboles de Probabilidad y Bayes",
    "Variable Aleatoria",
    "Variables Aleatorias Conjuntas",
    "Distribuciones Probabilísticas",
    "Distribuciones de muestreo",
    "Intervalos de confianza",
    "Pruebas de hipótesis",
    "Correlación",
    "Regresión Lineal",
    "Base de datos",
    "Fórmulas",
    "Historial de Cálculos"
]

st.sidebar.title("📊 Menú principal")
categoria = st.sidebar.radio("Sección", MENU)

st.markdown("<div class='main-title'>Solver Estadístico Unificado</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Solver Estadístico es mucho más que una calculadora: es una herramienta de apoyo operativo diseñada para acompañar al estudiante en cada etapa del aprendizaje estadístico. Desde la organización e interpretación de datos hasta el desarrollo de análisis inferenciales, transforma procesos complejos en experiencias claras, guiadas paso a paso y fáciles de comprender.</div>",
    unsafe_allow_html=True
)

# ============================================================
# INICIO
# ============================================================

if categoria == "Inicio":
    render_section_header(
        "Solver Estadístico Unificado",
        "Plataforma unificada para análisis estadístico, muestreo, hipótesis, conteo, correlación y exploración de datos.",
        "🏠",
        show_chip=False
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "<div class='metric-card'><h3>🧠 Módulo Probabilidad y Estadística</h3><p>Tablas de frecuencia, indicadores, probabilidad y Venn, conteo, árboles interactivos y Bayes, variables aleatorias (discretas, continuas y conjuntas) y distribuciones.</p></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            "<div class='metric-card'><h3>📘 Módulo Estadística Aplicada</h3><p>Distribuciones de muestreo y TLC, intervalos de confianza, pruebas de hipótesis con región de rechazo, correlación, regresión simple y múltiple.</p></div>",
            unsafe_allow_html=True
        )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(
            "<div class='metric-card'><h3>📂 Trabajo con archivos</h3><p>Puedes subir archivos CSV o Excel para análisis, correlaciones, exploración básica e indicadores con gráficos.</p></div>",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            "<div class='metric-card'><h3>📝 Salidas para informe</h3><p>El sistema genera conclusiones descargables y resultados listos para usar en trabajos académicos.</p></div>",
            unsafe_allow_html=True
        )

    st.info("Usa el menú lateral para navegar por todos los módulos del sistema.")

# ============================================================
# DISTRIBUCIONES DE MUESTREO
# ============================================================

elif categoria == "Distribuciones de muestreo":
    render_section_header(
        "Distribuciones de muestreo",
        "Media muestral y proporción muestral con una estructura uniforme de entrada y salida.",
        "📌"
    )
    tema = render_topic_selector(["Media muestral", "Proporción muestral", "Simulación del Teorema del Límite Central"], "tema_dist_muest")

    if tema == "Simulación del Teorema del Límite Central":
        distribuciones.render_tlc()
    elif tema == "Media muestral":
        st.markdown("### Datos de entrada")
        mu, sigma, n = render_three_inputs([
            ("μ media poblacional", 50),
            ("σ desviación poblacional", 10),
            ("n tamaño de muestra", 36)
        ])
        n = int(n)

        tipo = st.selectbox("Tipo de cálculo", ["P(X̄ < c)", "P(X̄ > c)", "P(a < X̄ < b)"], key="dm_media_tipo")

        if tipo in ["P(X̄ < c)", "P(X̄ > c)"]:
            c = parse_num(st.text_input("Media muestral (x̄)", "52", key="dm_media_c"))
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                a = parse_num(st.text_input("Límite inferior a", "45", key="dm_media_a"))
            with col_b:
                b = parse_num(st.text_input("Límite superior b", "55", key="dm_media_b"))

        se = sigma / math.sqrt(n) if n > 0 else float('inf')
        col_formula, col_se = st.columns(2)
        with col_formula:
            st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
        with col_se:
            render_result_box("Error estándar", f"{se:.6f}")

        if tipo in ["P(X̄ < c)", "P(X̄ > c)"]:
            if st.button("Calcular", key="dm_media_btn_1"):
                z = (c - mu) / se
                p_menor = phi(z)
                st.latex(r"z = \frac{\bar{x}-\mu}{SE}")
                st.write(f"z = {z:.6f}")
                if tipo == "P(X̄ < c)":
                    st.success(f"P(X̄ < {c}) = {fmt_prob(p_menor)}")
                    show_report(f"La probabilidad de que la media muestral sea menor que {c} es aproximadamente {fmt_prob(p_menor)}.")
                else:
                    prob = 1 - p_menor
                    st.success(f"P(X̄ > {c}) = {fmt_prob(prob)}")
                    show_report(f"La probabilidad de que la media muestral sea mayor que {c} es aproximadamente {fmt_prob(prob)}.")
        else:
            if st.button("Calcular", key="dm_media_btn_2"):
                z_a = (a - mu) / se
                z_b = (b - mu) / se
                prob = phi(z_b) - phi(z_a)
                st.latex(r"z_a = \frac{a-\mu}{SE}, \quad z_b = \frac{b-\mu}{SE}")
                st.write(f"z(a) = {z_a:.6f}")
                st.write(f"z(b) = {z_b:.6f}")
                st.success(f"P({a} < X̄ < {b}) = {fmt_prob(prob)}")
                show_report(f"La probabilidad de que la media muestral esté entre {a} y {b} es aproximadamente {fmt_prob(prob)}.")

    else:
        st.markdown("### Datos de entrada")
        p, n = render_three_inputs([
            ("p proporción poblacional", "17%"),
            ("n tamaño de muestra", 400),
            ("Campo auxiliar", 0)
        ])[:2]
        n = int(n)

        tipo = st.selectbox("Tipo de cálculo", ["P(p̂ < c)", "P(p̂ > c)", "P(a < p̂ < b)"], key="dm_prop_tipo")

        if tipo in ["P(p̂ < c)", "P(p̂ > c)"]:
            c = parse_num(st.text_input("Proporción muestral (p̂)", "20%", key="dm_prop_c"))
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                a = parse_num(st.text_input("Límite inferior a", "10%", key="dm_prop_a"))
            with col_b:
                b = parse_num(st.text_input("Límite superior b", "20%", key="dm_prop_b"))

        se = math.sqrt(p * (1 - p) / n) if n > 0 and 0 <= p <= 1 else float('inf')
        col_formula, col_se = st.columns(2)
        with col_formula:
            st.latex(r"SE = \sqrt{\frac{p(1-p)}{n}}")
        with col_se:
            render_result_box("Error estándar", f"{se:.6f}")
        st.write(f"n = {n}")
        st.write(f"n·p = {n*p:.4f}")
        if n >= 30 and n * p >= 5 and n * (1 - p) >= 5:
            st.success("Se cumple la aproximación normal usada normalmente en clase.")
        else:
            st.warning("No se cumple completamente la regla n ≥ 30, n·p ≥ 5 y n(1-p) ≥ 5.")

        if tipo in ["P(p̂ < c)", "P(p̂ > c)"]:
            if st.button("Calcular", key="dm_prop_btn_1"):
                z = (c - p) / se
                p_menor = phi(z)
                st.latex(r"z = \frac{\hat{p}-p}{SE}")
                st.write(f"z = {z:.6f}")
                if tipo == "P(p̂ < c)":
                    st.success(f"P(p̂ < {c:.4f}) = {fmt_prob(p_menor)}")
                    show_report(f"La probabilidad de que la proporción muestral sea menor que {c:.4f} es aproximadamente {fmt_prob(p_menor)}.")
                else:
                    prob = 1 - p_menor
                    st.success(f"P(p̂ > {c:.4f}) = {fmt_prob(prob)}")
                    show_report(f"La probabilidad de que la proporción muestral sea mayor que {c:.4f} es aproximadamente {fmt_prob(prob)}.")
        else:
            if st.button("Calcular", key="dm_prop_btn_2"):
                z_a = (a - p) / se
                z_b = (b - p) / se
                prob = phi(z_b) - phi(z_a)
                st.latex(r"z_a = \frac{a-p}{SE}, \quad z_b = \frac{b-p}{SE}")
                st.write(f"z(a) = {z_a:.6f}")
                st.write(f"z(b) = {z_b:.6f}")
                st.success(f"P({a:.4f} < p̂ < {b:.4f}) = {fmt_prob(prob)}")
                show_report(f"La probabilidad de que la proporción muestral esté entre {a:.4f} y {b:.4f} es aproximadamente {fmt_prob(prob)}.")

# ============================================================
# INTERVALOS DE CONFIANZA
# ============================================================

elif categoria == "Intervalos de confianza":
    render_section_header(
        "Intervalos de confianza",
        "Cálculo de intervalos para media (Z/t) y proporción con lógica automática o manual.",
        "📏"
    )
    tema = render_topic_selector(["Media (Una muestra)", "Proporción (Una muestra)"] + inferencia.TEMAS_IC_EXTRA, "tema_ic")

    if tema in inferencia.TEMAS_IC_EXTRA:
        inferencia.render_ic_extra(tema)
    elif tema == "Media (Una muestra)":
        st.markdown("### Datos de entrada")
        c1, c2, c3_inp = st.columns(3)
        with c1:
            nc = parse_num(st.text_input("Nivel de confianza", "95%", key="ic_media_nc"))
            if nc > 1: nc /= 100
        with c2:
            n = int(parse_num(st.text_input("Tamaño de muestra (n)", "30", key="ic_media_n")))
        with c3_inp:
            xbar = parse_num(st.text_input("Media muestral (x̄)", "100", key="ic_media_xbar"))

        col_met, col_vcrit = st.columns(2)
        with col_met:
            seleccion = st.radio("Método de cálculo", ["Automático", "Z (σ conocida)", "t (σ desconocida)"], horizontal=True, key="ic_media_metodo")
        with col_vcrit:
            alpha_prev = 1 - nc
            if seleccion == "Automático":
                if n >= 30:
                    crit_prev = abs(inv_norm(1 - alpha_prev / 2))
                    nombre_crit_prev = "z*"
                else:
                    crit_prev = critico_t(alpha_prev, n - 1, "Bilateral")
                    nombre_crit_prev = f"t* (gl={n-1})"
            elif seleccion == "Z (σ conocida)":
                crit_prev = abs(inv_norm(1 - alpha_prev / 2))
                nombre_crit_prev = "z*"
            else:
                crit_prev = critico_t(alpha_prev, n - 1, "Bilateral")
                nombre_crit_prev = f"t* (gl={n-1})"
            st.metric(f"Valor crítico ({nombre_crit_prev})", f"{crit_prev:.4f}")

        label_desv = "Desviación estándar (σ)" if seleccion == "Z (σ conocida)" else "Desviación estándar (s)"
        desv = parse_num(st.text_input(label_desv, "15", key="ic_media_desv"))
        
        contexto = st.text_input("Contexto (opcional)", "la variable en estudio", key="ic_media_ctx")

        if st.button("Calcular intervalo", key="ic_media_btn"):
            alpha = 1 - nc
            if seleccion == "Automático":
                if n >= 30:
                    crit = abs(inv_norm(1 - alpha / 2))
                    razon = f"Se utiliza Z como aproximación porque n={n} ≥ 30."
                    nombre_crit = "z*"
                else:
                    crit = critico_t(alpha, n - 1, "Bilateral")
                    razon = f"Se utiliza t de Student porque n={n} < 30 y se asume σ desconocida."
                    nombre_crit = f"t* (gl={n-1})"
            elif seleccion == "Z (σ conocida)":
                crit = abs(inv_norm(1 - alpha / 2))
                razon = "Se utiliza Z porque la desviación estándar poblacional (σ) es conocida."
                nombre_crit = "z*"
            else:
                crit = critico_t(alpha, n - 1, "Bilateral")
                razon = f"Se utiliza t de Student con {n-1} grados de libertad."
                nombre_crit = f"t* (gl={n-1})"

            se = desv / math.sqrt(n)
            margen = crit * se
            li, ls = xbar - margen, xbar + margen
            
            st.info(razon)
            st.latex(r"IC = \bar{x} \pm " + nombre_crit + r" \cdot \frac{s}{\sqrt{n}}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Valor crítico ({nombre_crit})", f"{crit:.4f}")
            c2.metric("Error estándar", f"{se:.4f}")
            c3.metric("Margen de error", f"{margen:.4f}")
            
            st.success(f"IC {nc*100:.2f}% = ({li:.4f}, {ls:.4f})")
            inferencia.grafico_intervalo(li, ls, xbar, "Intervalo de confianza para μ", "ic_media_g")
            
            reporte = (f"Con un nivel de confianza del {nc*100:.2f}%, el intervalo de confianza para la media poblacional de "
                      f"{contexto} se encuentra entre {li:.4f} y {ls:.4f}. Esto indica que, si repitiéramos el muestreo "
                      f"múltiples veces, el {nc*100:.0f}% de los intervalos calculados contendrían el valor real del parámetro.")
            show_report(reporte)

    else:
        st.markdown("### Datos de entrada")
        nc = parse_num(st.text_input("Nivel de confianza", "95%", key="ic_prop_nc"))
        if nc > 1: nc /= 100

        modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"], horizontal=True, key="ic_prop_modo")

        c1, c2 = st.columns(2)
        if modo == "Ya tengo p̂":
            with c1: phat = parse_num(st.text_input("Proporción muestral (p̂)", "0.50", key="ic_prop_phat"))
            with c2: n = int(parse_num(st.text_input("Tamaño de muestra (n)", "400", key="ic_prop_n")))
        else:
            with c1: x = parse_num(st.text_input("Número de éxitos (x)", "200", key="ic_prop_x"))
            with c2: n = int(parse_num(st.text_input("Tamaño de muestra (n)", "400", key="ic_prop_n")))
            phat = x / n
        
        contexto = st.text_input("Contexto (opcional)", "la proporción en estudio", key="ic_prop_ctx")

        if st.button("Calcular intervalo", key="ic_prop_btn"):
            alpha = 1 - nc
            z = abs(inv_norm(1 - alpha / 2))
            se = math.sqrt(phat * (1 - phat) / n) if n > 0 and 0 <= phat <= 1 else float('inf')
            margen = z * se
            li, ls = phat - margen, phat + margen
            
            st.latex(r"IC = \hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
            
            st.write(f"p̂ = {phat:.6f}")
            st.write(f"z* = {z:.4f}")
            st.write(f"SE = {se:.6f}")
            st.success(f"IC {nc*100:.2f}% = ({li:.6f}, {ls:.6f})")
            inferencia.grafico_intervalo(li, ls, phat, "Intervalo de confianza para p", "ic_prop_g")
            
            reporte = (f"Con un nivel de confianza del {nc*100:.2f}%, el intervalo de confianza para la proporción poblacional de "
                      f"{contexto} se encuentra entre {li:.4f} y {ls:.4f} (o entre {li*100:.2f}% y {ls*100:.2f}%).")
            show_report(reporte)

# ============================================================
# PRUEBAS DE HIPÓTESIS
# ============================================================

elif categoria == "Pruebas de hipótesis":
    render_section_header(
        "Pruebas de hipótesis",
        "Todas las pruebas comparten la misma estructura visual: tema, datos, tipo de prueba, cálculo y conclusión.",
        "🧪"
    )
    tema = render_topic_selector(["Media con Z", "Media con t", "Proporción con Z", "Diferencia de medias", "Diferencia de proporciones", "Varianza con χ²"], "tema_ph")

    if tema == "Varianza con χ²":
        inferencia.render_prueba_varianza()
    elif tema in ["Media con Z", "Media con t"]:
        mu0, xbar, desv, n = render_four_inputs([
            ("μ₀", 50),
            ("x̄", 53),
            (("σ poblacional" if tema == "Media con Z" else "s muestral"), 12),
            ("n", 36 if tema == "Media con Z" else 25)
        ])
        n = int(n)
        alpha = parse_num(st.text_input("α", "0.05", key=f"ph_alpha_{tema}"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key=f"ph_tipo_{tema}")
        contexto = st.text_area("Contexto para la conclusión", "la media observada debe analizarse frente al valor de referencia planteado.", key=f"ph_ctx_{tema}")

        if st.button("Resolver prueba", key=f"ph_btn_{tema}"):
            se = desv / math.sqrt(n) if n > 0 else float('inf')
            estad = (xbar - mu0) / se
            if tema == "Media con Z":
                p_value = p_valor_normal(estad, tipo)
                crit = critico_z(alpha, tipo)
                nombre = "z"
                gl_txt = "No aplica"
            else:
                gl = n - 1
                p_value = p_valor_t(estad, gl, tipo)
                crit = critico_t(alpha, gl, tipo)
                nombre = "t"
                gl_txt = str(gl)
            rechaza = p_value < alpha
            st.write(f"gl = {gl_txt}")
            st.write(f"SE = {se:.6f}")
            st.write(f"{nombre} calculado = {estad:.6f}")
            st.write(f"{nombre} crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            st.write(f"α = {alpha:.4f}")
            if rechaza:
                st.success("Decisión: Se rechaza H₀.")
            else:
                st.warning("Decisión: No se rechaza H₀.")
            inferencia.grafico_region_rechazo(estad, tipo, alpha, "z" if tema == "Media con Z" else "t", None if tema == "Media con Z" else n - 1, key="ph_media_rr")
            h1 = (
                "la media poblacional es diferente al valor de referencia"
                if tipo == "Bilateral"
                else ("la media poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la media poblacional es mayor al valor de referencia")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    elif tema == "Proporción con Z":
        p0 = parse_num(st.text_input("p₀", "17%", key="ph_prop_p0"))
        alpha = parse_num(st.text_input("α", "0.05", key="ph_prop_alpha"))
        modo = st.radio("¿Cómo te dieron la información?", ["Ya tengo p̂", "Tengo x y n"], horizontal=True, key="ph_prop_modo")
        if modo == "Ya tengo p̂":
            phat, n = render_three_inputs([
                ("p̂", 0.20),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
        else:
            x, n = render_three_inputs([
                ("x éxitos", 80),
                ("n", 400),
                ("Campo auxiliar", 0)
            ])[:2]
            phat = x / n
        n = int(n)
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="ph_prop_tipo")
        contexto = st.text_area("Contexto para la conclusión", "la proporción observada debe compararse con el porcentaje establecido como referencia.", key="ph_prop_ctx")

        if st.button("Resolver prueba", key="ph_prop_btn"):
            se = math.sqrt(p0 * (1 - p0) / n) if n > 0 and 0 <= p0 <= 1 else float('inf')
            z = (phat - p0) / se
            p_value = p_valor_normal(z, tipo)
            crit = critico_z(alpha, tipo)
            rechaza = p_value < alpha
            st.write(f"p̂ = {phat:.6f}")
            st.write(f"SE = {se:.6f}")
            st.write(f"z calculado = {z:.6f}")
            st.write(f"z crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            if rechaza:
                st.success("Decisión: Se rechaza H₀.")
            else:
                st.warning("Decisión: No se rechaza H₀.")
            inferencia.grafico_region_rechazo(z, tipo, alpha, "z", key="ph_prop_rr")
            h1 = (
                "la proporción poblacional es diferente al valor de referencia"
                if tipo == "Bilateral"
                else ("la proporción poblacional es menor al valor de referencia" if tipo == "Cola izquierda" else "la proporción poblacional es mayor al valor de referencia")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    elif tema == "Diferencia de medias":
        metodo = st.selectbox("Método", ["Muestras grandes con Z", "t pooled varianzas iguales", "t Welch varianzas diferentes", "Muestras pareadas"], key="ph_difm_met")
        alpha = parse_num(st.text_input("α", "0.05", key="difm_alpha"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="difm_tipo")
        d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0", key="difm_d0"))
        contexto = st.text_area("Contexto para la conclusión", "los promedios de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.", key="difm_ctx")

        if metodo != "Muestras pareadas":
            st.markdown("### Datos de entrada")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Grupo 1")
                xbar1 = parse_num(st.text_input("x̄₁", "42.48", key="difm_x1"))
                s1 = parse_num(st.text_input("s₁", "7.19", key="difm_s1"))
                n1 = int(parse_num(st.text_input("n₁", "205", key="difm_n1")))
            with c2:
                st.markdown("#### Grupo 2")
                xbar2 = parse_num(st.text_input("x̄₂", "43.04", key="difm_x2"))
                s2 = parse_num(st.text_input("s₂", "7.47", key="difm_s2"))
                n2 = int(parse_num(st.text_input("n₂", "195", key="difm_n2")))

            if st.button("Resolver prueba", key="dif_medias"):
                if metodo == "Muestras grandes con Z":
                    se = math.sqrt((s1**2 / n1) + (s2**2 / n2)) if n1 > 0 and n2 > 0 else float('inf')
                    estad = ((xbar1 - xbar2) - d0) / se if se != 0 else float('inf')
                    p_value = p_valor_normal(estad, tipo) if se != float('inf') else 1.0
                    crit = critico_z(alpha, tipo)
                    nombre = "z"
                    gl_txt = "No aplica"
                elif metodo == "t pooled varianzas iguales":
                    gl = n1 + n2 - 2
                    sp2 = (((n1 - 1) * s1**2) + ((n2 - 1) * s2**2)) / gl if gl > 0 else 0
                    sp = math.sqrt(sp2)
                    se = sp * math.sqrt((1/n1) + (1/n2)) if n1 > 0 and n2 > 0 else float('inf')
                    estad = ((xbar1 - xbar2) - d0) / se if se != 0 else float('inf')
                    p_value = p_valor_t(estad, gl, tipo) if gl > 0 and se != float('inf') else 1.0
                    crit = critico_t(alpha, gl, tipo) if gl > 0 else 0
                    nombre = "t"
                    gl_txt = str(gl)
                else:
                    se = math.sqrt((s1**2/n1) + (s2**2/n2)) if n1 > 0 and n2 > 0 else float('inf')
                    estad = ((xbar1 - xbar2) - d0) / se if se != 0 else float('inf')
                    num = ((s1**2/n1) + (s2**2/n2))**2 if n1 > 0 and n2 > 0 else 0
                    den = ((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1) if n1 > 1 and n2 > 1 else 0
                    gl = num / den if den > 0 else 0
                    p_value = p_valor_t(estad, gl, tipo) if gl > 0 and se != float('inf') else 1.0
                    crit = critico_t(alpha, gl, tipo) if gl > 0 else 0
                    nombre = "t"
                    gl_txt = f"{gl:.4f}"
                rechaza = p_value < alpha
                st.write(f"SE = {se:.6f}")
                st.write(f"{nombre} calculado = {estad:.6f}")
                st.write(f"gl = {gl_txt}")
                st.write(f"{nombre} crítico = {crit:.4f}")
                st.write(f"p-valor = {p_value:.6f}")
                if rechaza:
                    st.success("Decisión: Se rechaza H₀.")
                else:
                    st.warning("Decisión: No se rechaza H₀.")
                if gl_txt == "No aplica":
                    inferencia.grafico_region_rechazo(estad, tipo, alpha, "z", key="ph_difm_rr")
                elif gl > 0:
                    inferencia.grafico_region_rechazo(estad, tipo, alpha, "t", gl, key="ph_difm_rr")
                h1 = (
                    "existe diferencia entre las medias poblacionales"
                    if tipo == "Bilateral"
                    else ("la media del grupo 1 es menor que la media del grupo 2" if tipo == "Cola izquierda" else "la media del grupo 1 es mayor que la media del grupo 2")
                )
                show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))
        else:
            st.write("Ingresa datos separados por coma. La diferencia se calcula como después - antes.")
            antes_txt = st.text_area("Antes", "10,12,15,13,11", key="difm_antes")
            despues_txt = st.text_area("Después", "12,15,16,15,13", key="difm_despues")
            if st.button("Resolver prueba pareada", key="dif_pareada"):
                antes = np.array([parse_num(v) for v in antes_txt.split(",") if v.strip() != ""])
                despues = np.array([parse_num(v) for v in despues_txt.split(",") if v.strip() != ""])
                if len(antes) != len(despues):
                    st.error("Las dos listas deben tener la misma cantidad de datos.")
                elif len(antes) < 2:
                    st.error("Se necesitan mínimo dos pares de datos.")
                else:
                    d = despues - antes
                    n = len(d)
                    dbar = np.mean(d)
                    sd = np.std(d, ddof=1)
                    se = sd / math.sqrt(n) if n > 0 else float('inf')
                    gl = n - 1
                    estad = (dbar - d0) / se if se != 0 else float('inf')
                    p_value = p_valor_t(estad, gl, tipo) if gl > 0 and se != float('inf') else 1.0
                    crit = critico_t(alpha, gl, tipo) if gl > 0 else 0
                    rechaza = p_value < alpha
                    st.write(f"n = {n}")
                    st.write(f"Promedio de diferencias = {dbar:.6f}")
                    st.write(f"s_d = {sd:.6f}")
                    st.write(f"SE = {se:.6f}")
                    st.write(f"t calculado = {estad:.6f}")
                    st.write(f"gl = {gl}")
                    st.write(f"t crítico = {crit:.4f}")
                    st.write(f"p-valor = {p_value:.6f}")
                    if rechaza:
                        st.success("Decisión: Se rechaza H₀.")
                    else:
                        st.warning("Decisión: No se rechaza H₀.")
                    if gl > 0:
                        inferencia.grafico_region_rechazo(estad, tipo, alpha, "t", gl, key="ph_par_rr")
                    h1 = (
                        "existe diferencia significativa entre las mediciones pareadas"
                        if tipo == "Bilateral"
                        else ("la diferencia promedio es menor que el valor hipotético" if tipo == "Cola izquierda" else "la diferencia promedio es mayor que el valor hipotético")
                    )
                    show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

    else:
        alpha = parse_num(st.text_input("α", "0.05", key="difp_alpha"))
        tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="difp_tipo")
        d0 = parse_num(st.text_input("Diferencia hipotética d₀", "0", key="difp_d0"))
        contexto = st.text_area("Contexto para la conclusión", "las proporciones de los dos grupos deben compararse para establecer si existe una diferencia estadísticamente significativa.", key="difp_ctx")

        st.markdown("### Datos de entrada")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Grupo 1")
            x1 = int(parse_num(st.text_input("x₁ éxitos", "80", key="difp_x1")))
            n1 = int(parse_num(st.text_input("n₁", "205", key="difp_n1")))
        with c2:
            st.markdown("#### Grupo 2")
            x2 = int(parse_num(st.text_input("x₂ éxitos", "90", key="difp_x2")))
            n2 = int(parse_num(st.text_input("n₂", "195", key="difp_n2")))

        if st.button("Resolver prueba", key="dif_prop"):
            p1 = x1 / n1
            p2 = x2 / n2
            p_pool = (x1 + x2) / (n1 + n2)
            q_pool = 1 - p_pool
            se = math.sqrt(p_pool * q_pool * ((1/n1) + (1/n2)))
            z = ((p1 - p2) - d0) / se
            p_value = p_valor_normal(z, tipo)
            crit = critico_z(alpha, tipo)
            rechaza = p_value < alpha
            st.write(f"p̂₁ = {p1:.6f}")
            st.write(f"p̂₂ = {p2:.6f}")
            st.write(f"p combinada = {p_pool:.6f}")
            st.write(f"SE = {se:.6f}")
            st.write(f"z calculado = {z:.6f}")
            st.write(f"z crítico = {crit:.4f}")
            st.write(f"p-valor = {p_value:.6f}")
            if rechaza:
                st.success("Decisión: Se rechaza H₀.")
            else:
                st.warning("Decisión: No se rechaza H₀.")
            inferencia.grafico_region_rechazo(z, tipo, alpha, "z", key="ph_difp_rr")
            h1 = (
                "existe diferencia entre las proporciones poblacionales"
                if tipo == "Bilateral"
                else ("la proporción del grupo 1 es menor que la proporción del grupo 2" if tipo == "Cola izquierda" else "la proporción del grupo 1 es mayor que la proporción del grupo 2")
            )
            show_report(conclusion_hipotesis(rechaza, alpha, contexto, h1))

# ============================================================
# CORRELACIÓN
# ============================================================

elif categoria == "Correlación":
    render_section_header(
        "Correlación de Pearson",
        "Misma estructura para trabajo manual o con archivo.",
        "📈"
    )
    tema = render_topic_selector(["Datos manuales", "Archivo CSV o Excel"], "tema_corr")

    if tema == "Datos manuales":
        col1, col2 = st.columns(2)
        with col1:
            nombre_x = st.text_input("Nombre de X", "X")
            datos_x = st.text_area("Valores de X separados por coma", "1,2,3,4,5")
        with col2:
            nombre_y = st.text_input("Nombre de Y", "Y")
            datos_y = st.text_area("Valores de Y separados por coma", "2,4,6,8,10")

        if st.button("Calcular correlación", key="corr_manual"):
            try:
                x = np.array([parse_num(v) for v in datos_x.split(",") if v.strip() != ""])
                y = np.array([parse_num(v) for v in datos_y.split(",") if v.strip() != ""])
                if len(x) != len(y):
                    st.error("X y Y deben tener la misma cantidad de datos.")
                elif len(x) < 2:
                    st.error("Se necesitan mínimo 2 datos.")
                else:
                    r = np.corrcoef(x, y)[0, 1]
                    tipo_corr, intensidad = interpretar_correlacion(r)
                    st.success(f"Coeficiente de correlación r = {r:.4f}")
                    st.write(f"Correlación {tipo_corr} {intensidad} entre {nombre_x} y {nombre_y}.")
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.scatter(x, y)
                    ax.set_xlabel(nombre_x)
                    ax.set_ylabel(nombre_y)
                    ax.set_title(f"Diagrama de dispersión: {nombre_x} vs {nombre_y}")
                    st.pyplot(fig)
                    plt.close(fig)
                    if tipo_corr == "positiva":
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo_corr} {intensidad}. A medida que aumenta {nombre_x}, también tiende a aumentar {nombre_y}."
                    elif tipo_corr == "negativa":
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, lo que indica una correlación {tipo_corr} {intensidad}. A medida que aumenta {nombre_x}, {nombre_y} tiende a disminuir."
                    else:
                        reporte = f"El coeficiente de correlación de Pearson fue r = {r:.4f}, por lo que no se observa una relación lineal clara entre {nombre_x} y {nombre_y}."
                    show_report(reporte)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="corr_file")
        if archivo is not None:
            try:
                df = leer_csv_excel_cached(archivo.getvalue(), archivo.name.lower())
                st.success("Base cargada correctamente.")
                st.write(f"Filas: {df.shape[0]}")
                st.write(f"Columnas: {df.shape[1]}")
                st.dataframe(df.head())
                numericas = df.select_dtypes(include=["int64", "float64"])
                for col in ["ID", "Id", "id", "Registro", "registro"]:
                    if col in numericas.columns:
                        numericas = numericas.drop(columns=[col])
                if numericas.shape[1] < 2:
                    st.error("La base debe tener mínimo dos columnas numéricas.")
                else:
                    st.write("Variables numéricas utilizadas:")
                    st.write(list(numericas.columns))
                    matriz = numericas.corr(method="pearson")
                    st.subheader("Matriz de correlación")
                    st.dataframe(matriz.round(4))
                    plot_heatmap(matriz)
                    st.subheader("Interpretación automática")
                    textos = []
                    for i in range(len(matriz.columns)):
                        for j in range(i + 1, len(matriz.columns)):
                            var1, var2 = matriz.columns[i], matriz.columns[j]
                            r = matriz.iloc[i, j]
                            tipo_corr, intensidad = interpretar_correlacion(r)
                            linea = f"{var1} y {var2}: r = {r:.4f} → correlación {tipo_corr} {intensidad}."
                            st.write(linea)
                            textos.append(linea)
                    st.download_button(
                        "Descargar interpretación",
                        data="\n".join(textos),
                        file_name="interpretacion_correlaciones.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

# ============================================================
# BASE DE DATOS
# ============================================================

elif categoria == "Base de datos":
    render_section_header(
        "Base de datos",
        "Exploración básica o comparación por grupo con estructura uniforme de tema.",
        "📂"
    )
    tema = render_topic_selector(["Exploración básica", "Comparación por grupo"], "tema_bd")

    if tema == "Exploración básica":
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="bd_explo")
        if archivo is not None:
            try:
                df = leer_csv_excel_cached(archivo.getvalue(), archivo.name.lower())
                st.success("Base cargada correctamente.")
                st.write(f"Filas: {df.shape[0]}")
                st.write(f"Columnas: {df.shape[1]}")
                st.markdown("### Vista previa")
                st.dataframe(df.head())
                st.markdown("### Tipos de datos")
                st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipo"]))
                st.markdown("### Valores faltantes")
                faltantes = df.isna().sum().reset_index()
                faltantes.columns = ["Variable", "Faltantes"]
                st.dataframe(faltantes)
                numericas = df.select_dtypes(include=["int64", "float64"])
                if numericas.shape[1] > 0:
                    st.markdown("### Estadística descriptiva")
                    st.dataframe(numericas.describe().T)
                    variable = st.selectbox("Variable para histograma", numericas.columns)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(numericas[variable].dropna(), bins=20)
                    ax.set_title(f"Histograma de {variable}")
                    ax.set_xlabel(variable)
                    ax.set_ylabel("Frecuencia")
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="bd_grupo")
        if archivo is not None:
            try:
                df = leer_csv_excel_cached(archivo.getvalue(), archivo.name.lower())
                st.success("Base cargada correctamente.")
                st.dataframe(df.head())
                columnas = list(df.columns)
                col_grupo = st.selectbox("Variable de grupo", columnas)
                numericas = df.select_dtypes(include=["int64", "float64"]).columns
                col_num = st.selectbox("Variable numérica", numericas)
                if st.button("Comparar", key="comparar_bd"):
                    resumen = df.groupby(col_grupo)[col_num].agg(["count", "mean", "std", "min", "max"])
                    st.dataframe(resumen)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    grupos = [grupo[col_num].dropna().values for _, grupo in df.groupby(col_grupo)]
                    labels = [str(g) for g in df.groupby(col_grupo).groups.keys()]
                    ax.boxplot(grupos, tick_labels=labels)
                    ax.set_title(f"{col_num} por {col_grupo}")
                    ax.set_xlabel(col_grupo)
                    ax.set_ylabel(col_num)
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

# ============================================================
# FÓRMULAS
# ============================================================

elif categoria == "Fórmulas":
    render_section_header(
        "Formulario completo del programa",
        "Resumen de las fórmulas usadas en todos los cálculos del sistema.",
        "📚"
    )

    st.markdown("## 1. Distribuciones de muestreo")
    st.markdown("### Media muestral")
    st.latex(r"\mu_{\bar{x}} = \mu")
    st.latex(r"SE_{\bar{x}} = \frac{\sigma}{\sqrt{n}}")
    st.latex(r"z = \frac{\bar{x}-\mu}{\sigma/\sqrt{n}}")
    st.latex(r"P(\bar{X}<c), \quad P(\bar{X}>c), \quad P(a<\bar{X}<b)")

    st.markdown("### Proporción muestral")
    st.latex(r"\mu_{\hat{p}} = p")
    st.latex(r"SE_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}")
    st.latex(r"z = \frac{\hat{p}-p}{\sqrt{p(1-p)/n}}")
    st.latex(r"P(\hat{p}<c), \quad P(\hat{p}>c), \quad P(a<\hat{p}<b)")

    st.markdown("## 2. Intervalos de confianza")
    st.markdown("### Media con Z")
    st.latex(r"IC_\mu = \bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}")
    st.markdown("### Media con t")
    st.latex(r"IC_\mu = \bar{x} \pm t_{\alpha/2,\,n-1}\frac{s}{\sqrt{n}}")
    st.markdown("### Proporción con Z")
    st.latex(r"IC_p = \hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")

    st.markdown("## 3. Pruebas de hipótesis")
    st.markdown("### Media con Z")
    st.latex(r"z = \frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}")
    st.markdown("### Media con t")
    st.latex(r"t = \frac{\bar{x}-\mu_0}{s/\sqrt{n}}")
    st.latex(r"gl = n-1")
    st.markdown("### Proporción con Z")
    st.latex(r"z = \frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}")
    st.markdown("### Diferencia de medias con Z")
    st.latex(r"z = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}}")
    st.markdown("### Diferencia de medias con t (Welch)")
    st.latex(r"t = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}}")
    st.latex(r"gl \approx \frac{\left(\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}\right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1-1}+\frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2-1}}")
    st.markdown("### Diferencia de medias con t pooled")
    st.latex(r"s_p^2 = \frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}")
    st.latex(r"t = \frac{(\bar{x}_1-\bar{x}_2)-d_0}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}")
    st.latex(r"gl = n_1+n_2-2")
    st.markdown("### Muestras pareadas")
    st.latex(r"d_i = X_{después,i} - X_{antes,i}")
    st.latex(r"\bar{d} = \frac{\sum d_i}{n}")
    st.latex(r"t = \frac{\bar{d}-d_0}{s_d/\sqrt{n}}")
    st.latex(r"gl = n-1")
    st.markdown("### Diferencia de proporciones")
    st.latex(r"\hat{p}_1 = \frac{x_1}{n_1}, \quad \hat{p}_2 = \frac{x_2}{n_2}")
    st.latex(r"\hat{p} = \frac{x_1+x_2}{n_1+n_2}")
    st.latex(r"\hat{q} = 1-\hat{p}")
    st.latex(r"z = \frac{(\hat{p}_1-\hat{p}_2)-d_0}{\sqrt{\hat{p}\hat{q}\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")

    st.markdown("## 4. Correlación de Pearson")
    st.latex(r"r = \frac{\sum (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum (x_i-\bar{x})^2 \sum (y_i-\bar{y})^2}}")

    st.markdown("## 5. Indicadores estadísticos")
    st.latex(r"\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i")
    st.latex(r"\sigma^2 = \frac{\sum (x_i-\mu)^2}{N}")
    st.latex(r"s^2 = \frac{\sum (x_i-\bar{x})^2}{n-1}")
    st.latex(r"\sigma = \sqrt{\sigma^2}, \qquad s = \sqrt{s^2}")
    st.latex(r"CV = \frac{s}{\bar{x}} \times 100\%")
    st.latex(r"As = \frac{3(\bar{x}-Mo)}{s}")

    st.markdown("## 6. Reglas de conteo")
    st.latex(r"P(n)=n!")
    st.latex(r"P = \frac{n!}{n_1!n_2!\cdots n_k!}")
    st.latex(r"V(n,r)=\frac{n!}{(n-r)!}")
    st.latex(r"VR(n,r)=n^r")
    st.latex(r"C(n,r)=\binom{n}{r}=\frac{n!}{r!(n-r)!}")
    st.latex(r"CR(n,r)=\binom{n+r-1}{r}")

    st.markdown("## 7. Teorema de Bayes")
    st.latex(r"P(A_i|B)=\frac{P(A_i)P(B|A_i)}{\sum_{j=1}^{n}P(A_j)P(B|A_j)}")

    st.markdown("## 8. Variable aleatoria discreta")
    st.latex(r"E(X)=\sum x_i P(x_i)")
    st.latex(r"E(X^2)=\sum x_i^2 P(x_i)")
    st.latex(r"Var(X)=E(X^2)-[E(X)]^2")
    st.latex(r"\sigma_X = \sqrt{Var(X)}")

    st.markdown("## 9. Distribuciones probabilísticas")
    st.markdown("### Normal")
    st.latex(r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}")
    st.markdown("### Binomial")
    st.latex(r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}")
    st.markdown("### Poisson")
    st.latex(r"P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}")
    st.markdown("### Exponencial")
    st.latex(r"f(x)=\lambda e^{-\lambda x}, \quad x\geq 0")
    st.latex(r"P(X\le x)=1-e^{-\lambda x}")
    st.latex(r"P(X>x)=e^{-\lambda x}")
    st.markdown("### Hipergeométrica, Geométrica y Binomial negativa")
    st.latex(r"P(X=k)=\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}},\quad P(X=k)=(1-p)^{k-1}p,\quad P(X=k)=\binom{k-1}{r-1}p^r(1-p)^{k-r}")

    st.markdown("## 10. Tablas de frecuencia")
    st.latex(r"h_i=\frac{f_i}{n},\quad F_i=\sum_{j\le i} f_j,\quad H_i=\frac{F_i}{n}")
    st.latex(r"k = 1 + 3.322\log_{10} n,\qquad A=\frac{R}{k}")
    st.latex(r"\bar{x}=\frac{\sum x_i f_i}{n},\quad Me = L_i+\frac{n/2-F_{i-1}}{f_i}A,\quad Mo = L_i+\frac{d_1}{d_1+d_2}A")

    st.markdown("## 11. Probabilidad")
    st.latex(r"P(A\cup B)=P(A)+P(B)-P(A\cap B),\qquad P(A')=1-P(A)")
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)},\qquad P(A\cap B)=P(A)P(B\mid A)")
    st.latex(r"\text{Independencia: } P(A\cap B)=P(A)P(B)")
    st.latex(r"P(B)=\sum_i P(A_i)P(B\mid A_i)\quad\text{(probabilidad total)}")

    st.markdown("## 12. Variables aleatorias continuas y conjuntas")
    st.latex(r"\int_{-\infty}^{\infty} f(x)\,dx=1,\quad E(X)=\int x f(x)\,dx,\quad F(x)=\int_{-\infty}^{x} f(t)\,dt")
    st.latex(r"E(aX+b)=aE(X)+b,\qquad Var(aX+b)=a^2Var(X)")
    st.latex(r"Cov(X,Y)=E(XY)-E(X)E(Y),\qquad \rho=\frac{Cov(X,Y)}{\sigma_X\sigma_Y}")
    st.latex(r"Var(aX+bY)=a^2Var(X)+b^2Var(Y)+2ab\,Cov(X,Y)")

    st.markdown("## 13. Inferencia adicional")
    st.latex(r"IC_{\sigma^2}=\left(\frac{(n-1)s^2}{\chi^2_{\alpha/2}},\frac{(n-1)s^2}{\chi^2_{1-\alpha/2}}\right),\qquad \chi^2=\frac{(n-1)s^2}{\sigma_0^2}")
    st.latex(r"n=\left(\frac{z_{\alpha/2}\sigma}{E}\right)^2,\qquad n=\frac{z_{\alpha/2}^2p(1-p)}{E^2}")

    st.markdown("## 14. Regresión lineal")
    st.latex(r"\hat\beta_1=\frac{S_{xy}}{S_{xx}},\quad \hat\beta_0=\bar y-\hat\beta_1\bar x,\quad R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}")
    st.latex(r"\hat{\boldsymbol\beta}=(X^TX)^{-1}X^T\mathbf{y},\qquad R^2_{aj}=1-(1-R^2)\frac{n-1}{n-k-1},\qquad F=\frac{SSR/k}{SSE/(n-k-1)}")

# ============================================================
# MÓDULOS ACADÉMICOS ADICIONALES
# ============================================================

elif categoria == "Indicadores Estadísticos":
    render_section_header("Indicadores Estadísticos", "Tablero integral con medidas de tendencia, dispersión, forma y posición.", "🧮")
    fuente = st.radio("Fuente de datos", ["Datos manuales", "Archivo CSV o Excel"], horizontal=True, key="ind_fuente")

    datos = []
    nombre_var = "Variable"
    es_cualitativa = False
    
    if fuente == "Datos manuales":
        d_str = st.text_input("Ingrese datos separados por punto y coma", "12; 14; 10; 15", key="ind_manual_input")
        try:
            datos = [parse_num(x.strip()) for x in d_str.split(";") if x.strip()]
            nombre_var = "Datos Manuales"
            es_cualitativa = False
        except Exception:
            st.warning("Asegúrate de ingresar solo números válidos separados por punto y coma.")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="ind_file_input")
        if archivo is not None:
            try:
                df_ind = leer_csv_excel_cached(archivo.getvalue(), archivo.name.lower())
                with st.expander("Vista previa del archivo"):
                    st.dataframe(df_ind.head())
                
                todas_opciones = df_ind.columns.tolist()
                
                if not todas_opciones:
                    st.error("El archivo está vacío.")
                else:
                    nombre_var = st.selectbox("Selecciona la variable para analizar", todas_opciones, key="ind_col_select")
                    
                    # Determinar si la columna es cualitativa
                    # Intentamos la conversión robusta
                    serie_s = df_ind[nombre_var].astype(str).str.replace(',', '.', regex=False)
                    col_data_numeric = pd.to_numeric(serie_s, errors='coerce')
                    
                    total_no_nulos = df_ind[nombre_var].dropna().shape[0]
                    if total_no_nulos > 0 and (col_data_numeric.dropna().shape[0] / total_no_nulos) > 0.5:
                        es_cualitativa = False
                        col_data = col_data_numeric.dropna()
                        datos = col_data.tolist()
                    else:
                        es_cualitativa = True
                        datos = df_ind[nombre_var].dropna().astype(str).tolist()
                        
                    if not datos:
                        st.warning(f"La columna '{nombre_var}' no tiene datos válidos.")
            except Exception as e:
                st.error(f"Error al leer el archivo: {e}")

    if len(datos) >= 2:
        st.markdown("---")
        
        # 1. CÁLCULO DE INDICADORES
        if not es_cualitativa:
            datos_ordenados = sorted(datos)
            # Tendencia
            media = statistics.mean(datos)
            mediana = statistics.median(datos)
            modas = calcular_moda(datos)
            # Dispersión
            t_disp = st.radio("Cálculo de dispersión", ["Muestra", "Población"], horizontal=True, key="ind_disp_tipo")
            desv = statistics.pstdev(datos) if t_disp == "Población" else statistics.stdev(datos)
            varianza = desv**2
            cv = (desv / media * 100) if media != 0 else 0
            rango = max(datos) - min(datos)
            # Forma
            if len(modas) == 1 and list(Counter(datos).values()).count(1) != len(datos):
                asimetria = (3 * (media - modas[0])) / desv if desv != 0 else 0
            else:
                asimetria = (3 * (media - mediana)) / desv if desv != 0 else 0
            # Posición
            p25 = calcular_percentil(datos_ordenados, 25)
            p50 = calcular_percentil(datos_ordenados, 50)
            p75 = calcular_percentil(datos_ordenados, 75)
            p90 = calcular_percentil(datos_ordenados, 90)
            iqr = p75 - p25
            
            curt = sum(((x - media) / desv) ** 4 for x in datos) / len(datos) - 3 if desv != 0 else 0

            # 2. VISUALIZACIÓN DE MÉTRICAS CUANTITATIVAS
            st.subheader("📊 Métricas Estadísticas")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Tendencia Central**")
                st.metric("Media", f"{media:.4f}")
                st.metric("Mediana", f"{mediana:.4f}")
                st.metric("Moda(s)", ", ".join(f"{x:g}" for x in modas[:3]))
            with c2:
                st.markdown("**Dispersión**")
                st.metric("Desv. Estándar", f"{desv:.4f}")
                st.metric("Varianza", f"{varianza:.4f}")
                st.metric("CV", f"{cv:.2f}%")
            with c3:
                st.markdown("**Posición (Percentiles)**")
                st.metric("P25 (Q1)", f"{p25:.4f}")
                st.metric("P50 (Mediana)", f"{p50:.4f}")
                st.metric("P75 (Q3)", f"{p75:.4f}")
            with c4:
                st.markdown("**Forma y Otros**")
                st.metric("Asimetría", f"{asimetria:.4f}")
                st.metric("Curtosis", f"{curt:.4f}")
                st.metric("IQR", f"{iqr:.4f}")
                st.metric("Rango", f"{rango:.4f}")
                st.metric("N (Total)", f"{len(datos)}")

            # 3. VISUALIZACIÓN DE GRÁFICOS CUANTITATIVOS
            st.markdown("---")
            st.subheader("🖼️ Visualización Gráfica")
            st.info("A continuación se presentan tres tipos de gráficos para que elijas el que mejor se adapte a tu informe.")
            
            g1, g2 = st.columns(2)
            g3, g4 = st.columns(2)
            
            # Histograma
            with g1:
                st.markdown("#### Histograma")
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(datos, bins='auto', color='#38bdf8', edgecolor='white')
                ax_hist.set_title(f"Distribución de {nombre_var}")
                st.pyplot(fig_hist)
                plt.close(fig_hist)
            
            # Gráfico de Barras (Frecuencia)
            with g2:
                st.markdown("#### Gráfico de Barras")
                freq_data = Counter(datos)
                labels, values = zip(*sorted(freq_data.items()))
                fig_bar, ax_bar = plt.subplots()
                ax_bar.bar([str(l) for l in labels], values, color='#818cf8')
                ax_bar.set_title(f"Frecuencia de {nombre_var}")
                plt.xticks(rotation=45)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
                
            # Gráfico de Torta
            with g3:
                st.markdown("#### Gráfico de Torta")
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(values, labels=[str(l) for l in labels], autopct='%1.1f%%', startangle=90)
                ax_pie.set_title(f"Composición de {nombre_var}")
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            # Diagrama de Cajas (Box Plot)
            with g4:
                st.markdown("#### Diagrama de Cajas")
                fig_box, ax_box = plt.subplots()
                ax_box.boxplot(datos, vert=False, patch_artist=True, 
                               boxprops=dict(facecolor='#10b981', color='#059669'),
                               medianprops=dict(color='white'))
                ax_box.set_title(f"Box Plot de {nombre_var}")
                st.pyplot(fig_box)
                plt.close(fig_box)

            # 4. CONCLUSIÓN INTEGRAL CUANTITATIVA
            st.markdown("---")
            if st.button("Generar reporte integral", key="ind_report_btn"):
                tipo_as = "simétrica" if abs(asimetria) < 0.1 else ("positiva" if asimetria > 0 else "negativa")
                reporte = (f"Análisis estadístico para {nombre_var}: \n"
                          f"- La media es {media:.4f} con una mediana de {mediana:.4f}, lo que sugiere un centro de datos cercano a {media:.2f}.\n"
                          f"- La dispersión ({t_disp.lower()}) muestra una desviación estándar de {desv:.4f} (CV: {cv:.2f}%).\n"
                          f"- La distribución presenta una asimetría {tipo_as} ({asimetria:.4f}).\n"
                          f"- En cuanto a su posición, el 75% de los datos se encuentran por debajo de {p75:.4f}.")
                show_report(reporte)

        else:
            # Cualitativa
            modas = calcular_moda(datos)
            n_total = len(datos)
            n_unicos = len(set(datos))

            # 2. VISUALIZACIÓN DE MÉTRICAS CUALITATIVAS
            st.subheader("📊 Métricas Estadísticas")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Tendencia (Cualitativa)**")
                st.metric("Moda(s)", ", ".join(str(x) for x in modas[:3]))
            with c2:
                st.markdown("**Frecuencias**")
                st.metric("Categorías Únicas", f"{n_unicos}")
            with c3:
                st.markdown("**Muestra**")
                st.metric("N (Total)", f"{n_total}")

            # 3. VISUALIZACIÓN DE GRÁFICOS CUALITATIVOS
            st.markdown("---")
            st.subheader("🖼️ Visualización Gráfica")
            st.info("Dado que la variable es cualitativa, se habilitan únicamente el Gráfico de Barras y el Gráfico de Torta.")
            
            g1, g2 = st.columns(2)
            freq_data = Counter(datos)
            labels, values = zip(*sorted(freq_data.items()))
            
            # Gráfico de Barras (Frecuencia)
            with g1:
                st.markdown("#### Gráfico de Barras")
                fig_bar, ax_bar = plt.subplots()
                ax_bar.bar([str(l) for l in labels], values, color='#818cf8')
                ax_bar.set_title(f"Frecuencia de {nombre_var}")
                plt.xticks(rotation=45)
                st.pyplot(fig_bar)
                plt.close(fig_bar)
                
            # Gráfico de Torta
            with g2:
                st.markdown("#### Gráfico de Torta")
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(values, labels=[str(l) for l in labels], autopct='%1.1f%%', startangle=90)
                ax_pie.set_title(f"Composición de {nombre_var}")
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            # 4. CONCLUSIÓN INTEGRAL CUALITATIVA
            st.markdown("---")
            if st.button("Generar reporte integral", key="ind_report_btn"):
                reporte = (f"Análisis de frecuencia para la variable cualitativa {nombre_var}: \n"
                          f"- N total de observaciones válidas: {n_total}.\n"
                          f"- Se identificaron {n_unicos} categorías distintas.\n"
                          f"- La(s) categoría(s) más frecuente(s) (moda) es/son: {', '.join(str(x) for x in modas)}.")
                show_report(reporte)
    elif len(datos) == 1:
        st.info("Se necesitan al menos 2 datos para realizar el análisis estadístico.")

elif categoria == "Tabla de Frecuencias":
    render_section_header("Tabla de Frecuencias", "Datos no agrupados, agrupados en intervalos (Sturges) y cualitativos, con gráficos e indicadores.", "📋")
    frecuencias.render_tabla_frecuencias(leer_csv_excel_cached)

elif categoria == "Probabilidad Básica":
    render_section_header("Probabilidad Básica", "Operaciones con eventos, diagramas de Venn, tablas de contingencia y enfoque frecuentista.", "🎯")
    probabilidad.render_probabilidad()

elif categoria == "Reglas de Conteo":
    render_section_header("Reglas de Conteo", "Principio multiplicativo, permutaciones, variaciones y combinaciones con listado y árbol.", "🔢")
    conteo.render_conteo()

elif categoria == "Árboles de Probabilidad y Bayes":
    render_section_header("Árboles de Probabilidad y Teorema de Bayes", "Construye, grafica y demuestra: regla de la multiplicación, probabilidad total, condicional y Bayes.", "🌳")
    arboles.render_arboles()

elif categoria == "Variable Aleatoria":
    render_section_header("Variable Aleatoria", "Esperanza, varianza, función de distribución y propiedades (discretas y continuas).", "🎲")
    variables.render_variable_aleatoria()

elif categoria == "Variables Aleatorias Conjuntas":
    render_section_header("Variables Aleatorias Conjuntas Discretas", "Marginales, condicionales, covarianza, correlación e independencia.", "🧩")
    variables.render_conjuntas()

elif categoria == "Distribuciones Probabilísticas":
    render_section_header("Distribuciones Probabilísticas", "Binomial, Poisson, Hipergeométrica, Geométrica, Normal, Exponencial, t, χ², F y más.", "📈")
    distribuciones.render_distribuciones()

elif categoria == "Regresión Lineal":
    render_section_header("Regresión Lineal", "Modelo simple y múltiple: estimación, inferencia, ANOVA, residuos y predicción.", "📉")
    regresion.render_regresion(leer_csv_excel_cached)

elif categoria == "Historial de Cálculos":
    render_section_header("Historial de Cálculos", "Registro local de cálculos realizados en esta sesión.", "📜")
    
    if "historial" in st.session_state and st.session_state["historial"]:
        for reg in reversed(st.session_state["historial"]):
            with st.expander(f"🕒 {reg['fecha']} - {reg['modulo']}"):
                st.json(reg["inputs"])
                
        datos_json = json.dumps(st.session_state["historial"], indent=2)
        st.download_button(
            "Descargar historial (JSON)",
            data=datos_json,
            file_name="historial_calculos.json",
            mime="application/json"
        )
        
        if st.button("🗑️ Borrar historial"):
            st.session_state["historial"] = []
            st.rerun()
    else:
        st.info("No hay historial guardado.")
