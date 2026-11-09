"""CA3.1 y CA3.2 · Variables aleatorias discretas, continuas y conjuntas discretas."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import integrate, optimize

from .comunes import (COLOR_A, COLOR_ACENTO, COLOR_GRIS, COLOR_OK, COLOR_PRIMARIO, a_float, codigo_r, compilar_expr,
                      fmt_num, mostrar_plotly, reporte)


# ==================================================================
# DISCRETA
# ==================================================================

def render_variable_aleatoria():
    tema = st.selectbox("Tipo de variable", ["Discreta (tabla de distribución)", "Continua (función de densidad f(x))"], key="va_tema")
    if tema.startswith("Discreta"):
        _discreta()
    else:
        _continua()


def _leer_distribucion_discreta():
    tipo = st.radio("La segunda columna contiene", ["Probabilidades P(x)", "Frecuencias (se convierten a P = f/n)"],
                    horizontal=True, key="va_d_tipo")
    base = pd.DataFrame({"x": [0, 1, 2, 3], "P(x) o frecuencia": ["1/8", "3/8", "3/8", "1/8"]})
    st.caption("Edita la tabla: agrega o borra filas. Las probabilidades aceptan 0.25, 25% o 1/4.")
    tabla = st.data_editor(base, num_rows="dynamic", hide_index=True, key="va_d_ed",
                           column_config={"x": st.column_config.NumberColumn("x (valor)"),
                                          "P(x) o frecuencia": st.column_config.TextColumn("P(x) o frecuencia")})
    tabla = tabla.dropna()
    try:
        xs = [float(v) for v in tabla["x"]]
        ps = [a_float(v) for v in tabla["P(x) o frecuencia"]]
    except Exception:
        st.error("Revisa la tabla: hay valores no numéricos.")
        return None
    if len(xs) < 1:
        st.info("Agrega al menos un valor.")
        return None
    if any(p < 0 for p in ps):
        st.error("No se permiten probabilidades o frecuencias negativas.")
        return None
    s = pd.Series(ps, index=xs).groupby(level=0).sum().sort_index()
    if tipo.startswith("Frecuencias"):
        if s.sum() == 0:
            st.error("La suma de frecuencias es 0.")
            return None
        s = s / s.sum()
    else:
        suma = s.sum()
        if abs(suma - 1) > 1e-6:
            st.error(f"Las probabilidades suman {suma:.6f}; deben sumar 1. "
                     "(Si son frecuencias, cambia la opción de arriba.)")
            return None
    return s.index.to_numpy(dtype=float), s.to_numpy(dtype=float)


def _discreta():
    res = _leer_distribucion_discreta()
    if res is None:
        return
    x, p = res
    mu = float(np.sum(x * p))
    ex2 = float(np.sum(x ** 2 * p))
    var = max(ex2 - mu ** 2, 0.0)
    sigma = math.sqrt(var)
    F = np.cumsum(p)
    mediana = float(x[np.argmax(F >= 0.5 - 1e-12)])
    modas = x[np.isclose(p, p.max())]
    asim = float(np.sum((x - mu) ** 3 * p) / sigma ** 3) if sigma > 0 else 0.0

    df = pd.DataFrame({"x": x, "P(x)": p, "F(x) = P(X ≤ x)": F, "x·P(x)": x * p, "x²·P(x)": x ** 2 * p,
                       "(x−μ)²·P(x)": (x - mu) ** 2 * p})
    st.markdown("### 📋 Tabla de cálculo")
    total = {c: df[c].sum() for c in ["P(x)", "x·P(x)", "x²·P(x)", "(x−μ)²·P(x)"]}
    total.update({"x": "Σ", "F(x) = P(X ≤ x)": ""})
    st.dataframe(pd.concat([df.astype(object), pd.DataFrame([total])]).style.format(precision=4, na_rep=""), hide_index=True)

    c = st.columns(6)
    c[0].metric("E(X) = μ", fmt_num(mu))
    c[1].metric("E(X²)", fmt_num(ex2))
    c[2].metric("Var(X)", fmt_num(var))
    c[3].metric("σ", fmt_num(sigma))
    c[4].metric("Mediana", fmt_num(mediana))
    c[5].metric("Moda", ", ".join(fmt_num(m) for m in modas[:3]))
    st.caption(f"Coeficiente de variación = {fmt_num(sigma / mu * 100 if mu else float('nan'), 2)}% · "
               f"Asimetría = {fmt_num(asim)} ({'simétrica' if abs(asim) < 0.05 else 'sesgo a la derecha' if asim > 0 else 'sesgo a la izquierda'})")

    with st.expander("📐 Demostración paso a paso", expanded=True):
        n_max = 8
        terminos = " + ".join(f"({fmt_num(a)})({fmt_num(b)})" for a, b in list(zip(x, p))[:n_max])
        terminos2 = " + ".join(f"({fmt_num(a)})^2({fmt_num(b)})" for a, b in list(zip(x, p))[:n_max])
        if len(x) > n_max:
            terminos += r" + \cdots"
            terminos2 += r" + \cdots"
        st.latex(rf"E(X) = \sum x\,P(x) = {terminos} = {fmt_num(mu)}")
        st.latex(rf"E(X^2) = \sum x^2 P(x) = {terminos2} = {fmt_num(ex2)}")
        st.latex(rf"Var(X) = E(X^2) - [E(X)]^2 = {fmt_num(ex2)} - ({fmt_num(mu)})^2 = {fmt_num(var)}")
        st.latex(rf"Var(X) = \sum (x-\mu)^2 P(x) = {fmt_num(var)} \quad\text{{(forma equivalente)}}")
        st.latex(rf"\sigma = \sqrt{{Var(X)}} = \sqrt{{{fmt_num(var)}}} = {fmt_num(sigma)}")

    t1, t2 = st.tabs(["Función de probabilidad P(x)", "Función de distribución acumulada F(x)"])
    with t1:
        fig = go.Figure(go.Bar(x=x, y=p, marker_color=COLOR_PRIMARIO, text=[fmt_num(v) for v in p], textposition="outside", width=0.15 * (np.ptp(x) / max(len(x) - 1, 1) or 1)))
        fig.add_vrect(x0=mu - sigma, x1=mu + sigma, fillcolor=COLOR_ACENTO, opacity=0.12, line_width=0, annotation_text="μ ± σ")
        fig.add_vline(x=mu, line_dash="dash", line_color=COLOR_A, annotation_text=f"μ = {mu:.4f}")
        fig.update_layout(title="Función de probabilidad", xaxis_title="x", yaxis_title="P(X = x)")
        mostrar_plotly(fig, key="va_d_pmf")
    with t2:
        fig = go.Figure()
        xs_ext = np.concatenate([[x[0] - 1], x, [x[-1] + 1]])
        Fs_ext = np.concatenate([[0], F, [1]])
        for i in range(len(xs_ext) - 1):
            fig.add_scatter(x=[xs_ext[i], xs_ext[i + 1]], y=[Fs_ext[i], Fs_ext[i]], mode="lines", line_color=COLOR_OK, showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=F, mode="markers", marker=dict(color=COLOR_OK, size=9), name="cerrado", hovertemplate="F(%{x}) = %{y:.4f}<extra></extra>")
        fig.add_scatter(x=x, y=np.concatenate([[0], F[:-1]]), mode="markers", marker=dict(color="white", size=9, line=dict(color=COLOR_OK, width=2)), name="abierto", hoverinfo="skip")
        fig.update_layout(title="F(x) escalonada", xaxis_title="x", yaxis_title="F(x)", showlegend=False)
        mostrar_plotly(fig, key="va_d_cdf")

    st.markdown("### 🎯 Calcular probabilidades")
    c1, c2, c3, c4 = st.columns(4)
    inf = c1.selectbox("Límite inferior", ["sin límite", "X ≥ a", "X > a"], key="va_d_inf")
    a = c2.number_input("a", value=float(x[0]), key="va_d_a")
    sup = c3.selectbox("Límite superior", ["X ≤ b", "X < b", "sin límite"], key="va_d_sup")
    b = c4.number_input("b", value=float(mu), key="va_d_b")
    mask = np.ones_like(x, bool)
    partes = []
    if inf == "X ≥ a":
        mask &= x >= a; partes.append(f"X ≥ {fmt_num(a)}")
    elif inf == "X > a":
        mask &= x > a; partes.append(f"X > {fmt_num(a)}")
    if sup == "X ≤ b":
        mask &= x <= b; partes.append(f"X ≤ {fmt_num(b)}")
    elif sup == "X < b":
        mask &= x < b; partes.append(f"X < {fmt_num(b)}")
    prob = float(p[mask].sum())
    etiqueta = " y ".join(partes) or "todo el soporte"
    st.success(f"P({etiqueta}) = {prob:.6f} ({prob * 100:.2f}%)  →  suma de P(x) para x ∈ {{{', '.join(fmt_num(v) for v in x[mask])}}}")

    with st.expander("🔧 Propiedades de la esperanza: Y = aX + b y E[g(X)]"):
        c1, c2 = st.columns(2)
        aa = c1.number_input("a", value=2.0, key="va_d_ta")
        bb = c2.number_input("b", value=3.0, key="va_d_tb")
        st.latex(rf"E(aX+b) = aE(X)+b = ({fmt_num(aa)})({fmt_num(mu)}) + ({fmt_num(bb)}) = {fmt_num(aa * mu + bb)}")
        st.latex(rf"Var(aX+b) = a^2 Var(X) = ({fmt_num(aa)})^2({fmt_num(var)}) = {fmt_num(aa ** 2 * var)}")
        st.latex(rf"\sigma_{{aX+b}} = |a|\,\sigma = {fmt_num(abs(aa) * sigma)}")
        g_txt = st.text_input("g(x) =", "x**2 - 3*x + 1", key="va_d_g", help="Usa * para multiplicar y ** o ^ para potencias.")
        try:
            g = compilar_expr(g_txt)
            gx = np.array([float(g(x=v)) for v in x])
            eg = float(np.sum(gx * p))
            vg = float(np.sum(gx ** 2 * p) - eg ** 2)
            st.dataframe(pd.DataFrame({"x": x, "g(x)": gx, "P(x)": p, "g(x)·P(x)": gx * p}).style.format(precision=4), hide_index=True)
            st.latex(rf"E[g(X)] = \sum g(x)P(x) = {fmt_num(eg)}, \qquad Var[g(X)] = {fmt_num(vg)}")
        except Exception as e:
            st.error(f"No se pudo evaluar g(x): {e}")

    with st.expander("📏 Desigualdad de Chebyshev"):
        k = st.slider("k (número de desviaciones)", 1.1, 5.0, 2.0, 0.1, key="va_d_k")
        real = float(p[np.abs(x - mu) < k * sigma].sum()) if sigma > 0 else 1.0
        st.latex(rf"P(|X-\mu| < k\sigma) \ge 1 - \frac{{1}}{{k^2}} = {1 - 1 / k ** 2:.4f}")
        st.write(f"Valor real para esta distribución: P(|X − μ| < {k}σ) = **{real:.4f}** (siempre ≥ la cota).")

    with st.expander("Esperanza del índice i (versión anterior del módulo)"):
        i = np.arange(1, len(x) + 1)
        ei = float(np.sum(i * p)); vi = max(float(np.sum(i ** 2 * p)) - ei ** 2, 0)
        st.write(f"E(i) = {ei:.6f} · Var(i) = {vi:.6f} · σ(i) = {math.sqrt(vi):.6f}")

    codigo_r(f"""
x <- c({', '.join(fmt_num(v, 8) for v in x)})
p <- c({', '.join(fmt_num(v, 8) for v in p)})
sum(p)                       # debe ser 1
EX  <- sum(x * p)            # esperanza
EX2 <- sum(x^2 * p)
VarX <- EX2 - EX^2           # varianza
sdX <- sqrt(VarX)
Fx <- cumsum(p)              # distribución acumulada
c(EX = EX, VarX = VarX, sd = sdX)
barplot(p, names.arg = x, main = "P(X = x)")
plot(stepfun(x, c(0, Fx)), main = "F(x)", verticals = FALSE)
""")
    reporte(f"La variable aleatoria X tiene valor esperado E(X) = {fmt_num(mu)}, es decir, a largo plazo el promedio de sus valores "
            f"tiende a {fmt_num(mu)}. Su varianza es {fmt_num(var)} y su desviación estándar {fmt_num(sigma)}, lo que indica la "
            f"dispersión típica alrededor de la media. La probabilidad P({etiqueta}) es {prob:.4f}.", "va_d", "Variable aleatoria discreta")


# ==================================================================
# CONTINUA
# ==================================================================

def _lim(txt):
    t = str(txt).strip().lower().replace(" ", "")
    if t in ("inf", "+inf", "∞", "+∞", "infinito"):
        return math.inf
    if t in ("-inf", "-∞", "-infinito"):
        return -math.inf
    return a_float(t)


def _continua():
    st.caption("Escribe f(x) usando *, ** (o ^), sqrt, exp, log. Si la densidad tiene una constante desconocida, escríbela como k.")
    c1, c2, c3 = st.columns([2, 1, 1])
    expr = c1.text_input("f(x) =", "k*x**2", key="va_c_f")
    a_txt = c2.text_input("Desde a (puede ser -inf)", "0", key="va_c_a")
    b_txt = c3.text_input("Hasta b (puede ser inf)", "3", key="va_c_b")
    try:
        a, b = _lim(a_txt), _lim(b_txt)
        if not a < b:
            raise ValueError("a debe ser menor que b")
        tiene_k = "k" in expr.replace("sqrt", "")
        f_raw = compilar_expr(expr, variables=("x", "k"))
    except Exception as e:
        st.error(f"Revisa la entrada: {e}")
        return

    k = 1.0
    if tiene_k:
        area1, _ = integrate.quad(lambda t: float(f_raw(x=t, k=1.0)), a, b, limit=200)
        if area1 <= 0 or not math.isfinite(area1):
            st.error("No se pudo hallar k: la integral con k = 1 no es positiva y finita.")
            return
        k = 1 / area1
        st.markdown("#### Hallar la constante k")
        la0 = fmt_num(a) if math.isfinite(a) else r"-\infty"
        lb0 = fmt_num(b) if math.isfinite(b) else r"\infty"
        st.latex(rf"\int_{{{la0}}}^{{{lb0}}} k\cdot g(x)\,dx = 1")
        st.latex(rf"k\cdot{fmt_num(area1, 6)} = 1 \;\Rightarrow\; k = {fmt_num(k, 6)}")

    def f(t):
        return float(f_raw(x=t, k=k))

    try:
        area, _ = integrate.quad(f, a, b, limit=200)
        grid_a = a if math.isfinite(a) else -50
        grid_b = b if math.isfinite(b) else 50
        muestra = np.linspace(grid_a, grid_b, 400)
        if np.nanmin([f(t) for t in muestra]) < -1e-9:
            st.error("f(x) toma valores negativos en el intervalo: no es una función de densidad.")
            return
    except Exception as e:
        st.error(f"No se pudo integrar: {e}")
        return
    if abs(area - 1) > 1e-4:
        st.error(f"∫ f(x) dx = {area:.6f} ≠ 1. No es una densidad válida (usa k para normalizarla).")
        return

    mu = integrate.quad(lambda t: t * f(t), a, b, limit=200)[0]
    ex2 = integrate.quad(lambda t: t * t * f(t), a, b, limit=200)[0]
    var = max(ex2 - mu ** 2, 0)
    sigma = math.sqrt(var)

    def F(t):
        if t <= a:
            return 0.0
        return min(integrate.quad(f, a, min(t, b), limit=200)[0], 1.0)

    def cuantil(q):
        lo = a if math.isfinite(a) else mu - 50 * (sigma or 1)
        hi = b if math.isfinite(b) else mu + 50 * (sigma or 1)
        return optimize.brentq(lambda t: F(t) - q, lo, hi)

    mediana = cuantil(0.5)
    c = st.columns(5)
    c[0].metric("∫ f(x) dx", f"{area:.4f} ✅")
    c[1].metric("E(X)", fmt_num(mu))
    c[2].metric("Var(X)", fmt_num(var))
    c[3].metric("σ", fmt_num(sigma))
    c[4].metric("Mediana", fmt_num(mediana))

    la = fmt_num(a) if math.isfinite(a) else r"-\infty"
    lb = fmt_num(b) if math.isfinite(b) else r"\infty"
    with st.expander("📐 Demostración", expanded=True):
        st.latex(rf"E(X) = \int_{{{la}}}^{{{lb}}} x\,f(x)\,dx = {fmt_num(mu)}")
        st.latex(rf"E(X^2) = \int_{{{la}}}^{{{lb}}} x^2 f(x)\,dx = {fmt_num(ex2)}")
        st.latex(rf"Var(X) = E(X^2) - [E(X)]^2 = {fmt_num(ex2)} - ({fmt_num(mu)})^2 = {fmt_num(var)}")
        st.latex(rf"F(Me) = \int_{{{la}}}^{{Me}} f(x)\,dx = 0.5 \;\Rightarrow\; Me = {fmt_num(mediana)}")

    st.markdown("### 🎯 Probabilidad P(c ≤ X ≤ d)")
    c1, c2, c3 = st.columns(3)
    cc = c1.number_input("c", value=float(a if math.isfinite(a) else mu - sigma), key="va_c_c")
    dd = c2.number_input("d", value=float(mediana), key="va_c_d")
    q = c3.slider("Percentil a calcular", 1, 99, 90, key="va_c_q")
    prob = max(F(dd) - F(cc), 0) if dd >= cc else 0.0
    xq = cuantil(q / 100)
    st.success(f"P({fmt_num(cc)} ≤ X ≤ {fmt_num(dd)}) = ∫ f(x) dx = {prob:.6f} ({prob * 100:.2f}%)  ·  P{q} = {fmt_num(xq)}")
    st.caption("En una variable continua P(X = c) = 0, así que ≤ y < dan el mismo resultado.")

    lo = a if math.isfinite(a) else mu - 4 * sigma
    hi = b if math.isfinite(b) else mu + 4 * sigma
    xs = np.linspace(lo, hi, 400)
    ys = np.array([f(t) for t in xs])
    t1, t2 = st.tabs(["Densidad f(x)", "Acumulada F(x)"])
    with t1:
        fig = go.Figure()
        fig.add_scatter(x=xs, y=ys, mode="lines", line_color=COLOR_PRIMARIO, name="f(x)")
        zona = (xs >= cc) & (xs <= dd)
        fig.add_scatter(x=xs[zona], y=ys[zona], fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.35)", name=f"P = {prob:.4f}")
        fig.add_vline(x=mu, line_dash="dash", line_color=COLOR_OK, annotation_text="E(X)")
        fig.update_layout(title="Función de densidad", xaxis_title="x", yaxis_title="f(x)")
        mostrar_plotly(fig, key="va_c_pdf")
    with t2:
        xs2 = np.linspace(lo, hi, 120)
        fig = go.Figure(go.Scatter(x=xs2, y=[F(t) for t in xs2], mode="lines", line_color=COLOR_OK))
        fig.update_layout(title="Función de distribución F(x)", xaxis_title="x", yaxis_title="F(x)")
        mostrar_plotly(fig, key="va_c_cdf")

    r_a = fmt_num(a) if math.isfinite(a) else "-Inf"
    r_b = fmt_num(b) if math.isfinite(b) else "Inf"
    codigo_r(f"""
k <- {fmt_num(k, 10)}
f <- function(x) {expr.replace('**', '^')}
integrate(f, {r_a}, {r_b})                                   # debe dar 1
EX  <- integrate(function(x) x * f(x), {r_a}, {r_b})$value
EX2 <- integrate(function(x) x^2 * f(x), {r_a}, {r_b})$value
VarX <- EX2 - EX^2
integrate(f, {fmt_num(cc)}, {fmt_num(dd)})                   # P(c <= X <= d)
curve(f, {fmt_num(lo)}, {fmt_num(hi)}, main = "f(x)")
""")
    reporte(f"La función dada es una densidad válida{' con k = ' + fmt_num(k, 6) if tiene_k else ''}. La variable tiene media "
            f"E(X) = {fmt_num(mu)}, varianza {fmt_num(var)} y mediana {fmt_num(mediana)}. "
            f"La probabilidad de que X esté entre {fmt_num(cc)} y {fmt_num(dd)} es {prob:.4f}.", "va_c", "Variable aleatoria continua")


# ==================================================================
# CONJUNTAS DISCRETAS
# ==================================================================

def render_conjuntas():
    c1, c2, c3 = st.columns(3)
    xs_txt = c1.text_input("Valores de X (columnas)", "0, 1, 2", key="vc_x")
    ys_txt = c2.text_input("Valores de Y (filas)", "0, 1, 2", key="vc_y")
    tipo = c3.radio("Las celdas contienen", ["Probabilidades", "Frecuencias"], key="vc_tipo")
    try:
        xs = [a_float(v) for v in xs_txt.split(",") if v.strip()]
        ys = [a_float(v) for v in ys_txt.split(",") if v.strip()]
        if len(set(xs)) != len(xs) or len(set(ys)) != len(ys) or not xs or not ys:
            raise ValueError
    except Exception:
        st.error("Escribe valores numéricos distintos separados por coma.")
        return
    ejemplo = [["0.10", "0.04", "0.02"], ["0.08", "0.20", "0.06"], ["0.06", "0.14", "0.30"]]
    datos = [[ejemplo[i][j] if i < 3 and j < 3 and len(xs) == 3 and len(ys) == 3 else f"1/{len(xs) * len(ys)}"
              for j in range(len(xs))] for i in range(len(ys))]
    base = pd.DataFrame(datos, index=[f"y = {fmt_num(v)}" for v in ys], columns=[f"x = {fmt_num(v)}" for v in xs])
    st.markdown("#### Tabla conjunta p(x, y) — edita las celdas (acepta 0.1, 10%, 1/10)")
    tabla = st.data_editor(base, key=f"vc_ed_{xs_txt}_{ys_txt}")
    try:
        P = np.array([[a_float(v) for v in fila] for fila in tabla.values], dtype=float)
    except Exception:
        st.error("Hay celdas no numéricas.")
        return
    if (P < 0).any():
        st.error("No se permiten valores negativos.")
        return
    if tipo == "Frecuencias":
        P = P / P.sum()
    elif abs(P.sum() - 1) > 1e-6:
        st.error(f"Las probabilidades suman {P.sum():.6f}; deben sumar 1.")
        return
    X = np.array(xs)
    Y = np.array(ys)
    pX = P.sum(axis=0)
    pY = P.sum(axis=1)
    EX, EY = float(X @ pX), float(Y @ pY)
    VX = float((X ** 2) @ pX - EX ** 2)
    VY = float((Y ** 2) @ pY - EY ** 2)
    EXY = float(np.sum(np.outer(Y, X) * P))
    cov = EXY - EX * EY
    rho = cov / math.sqrt(VX * VY) if VX > 0 and VY > 0 else float("nan")
    indep_celdas = np.isclose(P, np.outer(pY, pX), atol=1e-9)
    indep = bool(indep_celdas.all())

    st.markdown("### 📋 Tabla con marginales")
    vista = pd.DataFrame(P, index=base.index, columns=base.columns)
    vista["p_Y(y)"] = pY
    vista.loc["p_X(x)"] = list(pX) + [P.sum()]
    st.dataframe(vista.style.format("{:.4f}"))

    c = st.columns(4)
    c[0].metric("E(X)", fmt_num(EX)); c[0].metric("Var(X)", fmt_num(VX))
    c[1].metric("E(Y)", fmt_num(EY)); c[1].metric("Var(Y)", fmt_num(VY))
    c[2].metric("E(XY)", fmt_num(EXY)); c[2].metric("Cov(X,Y)", fmt_num(cov))
    c[3].metric("ρ (correlación)", fmt_num(rho)); c[3].metric("¿Independientes?", "Sí ✅" if indep else "No ❌")

    with st.expander("📐 Demostración paso a paso", expanded=True):
        st.latex(r"p_X(x) = \sum_y p(x,y), \qquad p_Y(y) = \sum_x p(x,y)")
        st.latex(rf"E(X) = \sum x\,p_X(x) = " + " + ".join(f"({fmt_num(a)})({fmt_num(b)})" for a, b in zip(X, pX)) + f" = {fmt_num(EX)}")
        st.latex(rf"E(Y) = \sum y\,p_Y(y) = " + " + ".join(f"({fmt_num(a)})({fmt_num(b)})" for a, b in zip(Y, pY)) + f" = {fmt_num(EY)}")
        terms = [f"({fmt_num(xv)})({fmt_num(yv)})({fmt_num(P[i, j])})" for i, yv in enumerate(Y) for j, xv in enumerate(X) if P[i, j] and xv * yv]
        st.latex(r"E(XY) = \sum_x\sum_y xy\,p(x,y) = " + (" + ".join(terms[:9]) + (r" + \cdots" if len(terms) > 9 else "") if terms else "0") + f" = {fmt_num(EXY)}")
        st.latex(rf"Cov(X,Y) = E(XY) - E(X)E(Y) = {fmt_num(EXY)} - ({fmt_num(EX)})({fmt_num(EY)}) = {fmt_num(cov)}")
        st.latex(rf"\rho = \frac{{Cov(X,Y)}}{{\sigma_X\sigma_Y}} = \frac{{{fmt_num(cov)}}}{{\sqrt{{{fmt_num(VX)}}}\sqrt{{{fmt_num(VY)}}}}} = {fmt_num(rho)}")
        if indep:
            st.success("p(x, y) = p_X(x)·p_Y(y) en todas las celdas → X e Y son independientes.")
        else:
            i, j = np.argwhere(~indep_celdas)[0]
            st.warning(f"No son independientes: por ejemplo p({fmt_num(X[j])}, {fmt_num(Y[i])}) = {P[i, j]:.4f} ≠ "
                     f"p_X·p_Y = {pX[j]:.4f}·{pY[i]:.4f} = {pX[j] * pY[i]:.4f}")

    t1, t2, t3, t4 = st.tabs(["🗺️ Mapa de calor", "📊 3D", "🔀 Condicionales", "➕ Combinaciones y regiones"])
    with t1:
        fig = go.Figure(go.Heatmap(z=P, x=[fmt_num(v) for v in X], y=[fmt_num(v) for v in Y], colorscale="Blues",
                                   texttemplate="%{z:.3f}", hovertemplate="x=%{x}, y=%{y}<br>p=%{z:.4f}<extra></extra>"))
        fig.update_layout(title="p(x, y)", xaxis_title="X", yaxis_title="Y")
        mostrar_plotly(fig, key="vc_heat")
    with t2:
        fig = go.Figure()
        for i, yv in enumerate(Y):
            for j, xv in enumerate(X):
                fig.add_trace(go.Scatter3d(x=[xv, xv], y=[yv, yv], z=[0, P[i, j]], mode="lines", line=dict(width=12, color=COLOR_PRIMARIO), showlegend=False,
                                           hovertext=f"p({fmt_num(xv)}, {fmt_num(yv)}) = {P[i, j]:.4f}", hoverinfo="text"))
        fig.update_layout(title="Distribución conjunta (gira con el mouse)", height=520,
                          scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="p(x,y)"))
        mostrar_plotly(fig, key="vc_3d")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            y0 = st.selectbox("Condicionar en Y =", Y, format_func=fmt_num, key="vc_y0")
            i = list(Y).index(y0)
            if pY[i] > 0:
                cond = P[i] / pY[i]
                st.dataframe(pd.DataFrame({"x": X, f"P(X=x | Y={fmt_num(y0)})": cond}).style.format(precision=4), hide_index=True)
                st.latex(rf"E(X\mid Y={fmt_num(y0)}) = {fmt_num(float(X @ cond))}")
        with c2:
            x0 = st.selectbox("Condicionar en X =", X, format_func=fmt_num, key="vc_x0")
            j = list(X).index(x0)
            if pX[j] > 0:
                cond = P[:, j] / pX[j]
                st.dataframe(pd.DataFrame({"y": Y, f"P(Y=y | X={fmt_num(x0)})": cond}).style.format(precision=4), hide_index=True)
                st.latex(rf"E(Y\mid X={fmt_num(x0)}) = {fmt_num(float(Y @ cond))}")
    with t4:
        c1, c2 = st.columns(2)
        aa = c1.number_input("a", value=1.0, key="vc_a")
        bb = c2.number_input("b", value=1.0, key="vc_b")
        st.latex(rf"E(aX+bY) = aE(X)+bE(Y) = {fmt_num(aa * EX + bb * EY)}")
        st.latex(rf"Var(aX+bY) = a^2Var(X)+b^2Var(Y)+2ab\,Cov(X,Y) = {fmt_num(aa ** 2 * VX + bb ** 2 * VY + 2 * aa * bb * cov)}")
        st.markdown("**Probabilidad de una región** (usa x, y, <, <=, >, >=, ==, and, or)")
        cond_txt = st.text_input("Condición", "x + y <= 2", key="vc_cond")
        try:
            g = compilar_expr(cond_txt, variables=("x", "y"), logica=True)
            M = np.array([[bool(g(x=xv, y=yv)) for xv in X] for yv in Y])
            st.success(f"P({cond_txt}) = {P[M].sum():.6f}")
        except Exception as e:
            st.error(f"Condición no válida: {e}")
        z_txt = st.text_input("Distribución de Z = g(X, Y)", "x + y", key="vc_z")
        try:
            gz = compilar_expr(z_txt, variables=("x", "y"))
            dz = {}
            for i, yv in enumerate(Y):
                for j, xv in enumerate(X):
                    z = round(float(gz(x=xv, y=yv)), 10)
                    dz[z] = dz.get(z, 0) + P[i, j]
            dfz = pd.DataFrame(sorted(dz.items()), columns=["z", "P(Z = z)"])
            EZ = float((dfz["z"] * dfz["P(Z = z)"]).sum())
            st.dataframe(dfz.style.format(precision=4), hide_index=True)
            st.latex(rf"E(Z) = {fmt_num(EZ)}")
        except Exception as e:
            st.error(f"Expresión no válida: {e}")

    filas_r = ",\n              ".join(", ".join(fmt_num(v, 8) for v in fila) for fila in P)
    codigo_r(f"""
x <- c({', '.join(fmt_num(v) for v in X)})
y <- c({', '.join(fmt_num(v) for v in Y)})
P <- matrix(c({filas_r}), nrow = {len(Y)}, byrow = TRUE)   # filas = y, columnas = x
pX <- colSums(P); pY <- rowSums(P)                         # marginales
EX <- sum(x * pX); EY <- sum(y * pY)
EXY <- sum(outer(y, x) * P)
covXY <- EXY - EX * EY
rho <- covXY / sqrt((sum(x^2 * pX) - EX^2) * (sum(y^2 * pY) - EY^2))
all.equal(P, outer(pY, pX))                                # TRUE si son independientes
c(EX = EX, EY = EY, EXY = EXY, Cov = covXY, rho = rho)
""")
    reporte(f"Para la distribución conjunta dada, E(X) = {fmt_num(EX)} y E(Y) = {fmt_num(EY)}. La covarianza es {fmt_num(cov)} "
            f"y el coeficiente de correlación ρ = {fmt_num(rho)}, lo que indica una relación lineal "
            f"{'positiva' if cov > 0 else 'negativa' if cov < 0 else 'nula'} entre X e Y. "
            f"Las variables {'son' if indep else 'no son'} independientes.", "vc", "Variables conjuntas")
