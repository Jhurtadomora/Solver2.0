"""CA3.3 · Modelos de probabilidad especiales (discretos y continuos) + TLC."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from .comunes import COLOR_A, COLOR_ACENTO, COLOR_GRIS, COLOR_OK, COLOR_PRIMARIO, codigo_r, fmt_num, mostrar_plotly, reporte

# Cada distribución: parámetros (clave, etiqueta, tipo, defecto, min, max), constructor scipy,
# nombre en R + argumentos, desplazamiento para R, fórmulas de E y Var, descripción.
DISCRETAS = {
    "Binomial": dict(
        params=[("n", "Ensayos n", int, 10, 1, 100000), ("p", "Prob. de éxito p", float, 0.3, 0.0, 1.0)],
        build=lambda q: stats.binom(q["n"], q["p"]),
        r=lambda q: ("binom", f"size = {q['n']}, prob = {q['p']}", 0),
        E=r"E(X) = np", V=r"Var(X) = np(1-p)",
        f=r"P(X=k) = \binom{n}{k}p^k(1-p)^{n-k}",
        desc="X = número de éxitos en n ensayos independientes con probabilidad p constante.",
    ),
    "Poisson": dict(
        params=[("lam", "Tasa promedio λ", float, 4.0, 1e-9, 1e6)],
        build=lambda q: stats.poisson(q["lam"]),
        r=lambda q: ("pois", f"lambda = {q['lam']}", 0),
        E=r"E(X) = \lambda", V=r"Var(X) = \lambda",
        f=r"P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}",
        desc="X = número de ocurrencias en un intervalo (tiempo, área…) con tasa promedio λ.",
    ),
    "Hipergeométrica": dict(
        params=[("N", "Tamaño población N", int, 20, 1, 100000), ("K", "Éxitos en la población K", int, 7, 0, 100000),
                ("n", "Tamaño de muestra n", int, 5, 1, 100000)],
        build=lambda q: stats.hypergeom(q["N"], q["K"], q["n"]),
        r=lambda q: ("hyper", f"m = {q['K']}, n = {q['N'] - q['K']}, k = {q['n']}", 0),
        E=r"E(X) = n\frac{K}{N}", V=r"Var(X) = n\frac{K}{N}\left(1-\frac{K}{N}\right)\frac{N-n}{N-1}",
        f=r"P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}",
        desc="X = número de éxitos al extraer n elementos SIN reemplazo de una población de N con K éxitos.",
    ),
    "Geométrica": dict(
        params=[("p", "Prob. de éxito p", float, 0.2, 1e-9, 1.0)],
        build=lambda q: stats.geom(q["p"]),
        r=lambda q: ("geom", f"prob = {q['p']}", 1),
        E=r"E(X) = \frac{1}{p}", V=r"Var(X) = \frac{1-p}{p^2}",
        f=r"P(X=k) = (1-p)^{k-1}p,\quad k=1,2,\dots",
        desc="X = número del ensayo en el que ocurre el PRIMER éxito (k = 1, 2, …).",
    ),
    "Binomial negativa": dict(
        params=[("r", "Éxitos buscados r", int, 3, 1, 10000), ("p", "Prob. de éxito p", float, 0.4, 1e-9, 1.0)],
        build=lambda q: stats.nbinom(q["r"], q["p"], loc=q["r"]),
        r=lambda q: ("nbinom", f"size = {q['r']}, prob = {q['p']}", q["r"]),
        E=r"E(X) = \frac{r}{p}", V=r"Var(X) = \frac{r(1-p)}{p^2}",
        f=r"P(X=k) = \binom{k-1}{r-1}p^r(1-p)^{k-r},\quad k=r,r+1,\dots",
        desc="X = número del ensayo en el que se obtiene el r-ésimo éxito.",
    ),
    "Uniforme discreta": dict(
        params=[("a", "Mínimo a", int, 1, -100000, 100000), ("b", "Máximo b", int, 6, -100000, 100000)],
        build=lambda q: stats.randint(q["a"], q["b"] + 1),
        r=None,
        E=r"E(X) = \frac{a+b}{2}", V=r"Var(X) = \frac{(b-a+1)^2-1}{12}",
        f=r"P(X=k) = \frac{1}{b-a+1}",
        desc="Todos los enteros entre a y b son igualmente probables (p. ej. un dado).",
    ),
}

CONTINUAS = {
    "Normal": dict(
        params=[("mu", "Media μ", float, 0.0, -1e9, 1e9), ("sigma", "Desv. estándar σ", float, 1.0, 1e-9, 1e9)],
        build=lambda q: stats.norm(q["mu"], q["sigma"]),
        r=lambda q: ("norm", f"mean = {q['mu']}, sd = {q['sigma']}"),
        E=r"E(X) = \mu", V=r"Var(X) = \sigma^2",
        f=r"f(x) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
        desc="Distribución simétrica en forma de campana.",
    ),
    "Exponencial": dict(
        params=[("lam", "Tasa λ (1/media)", float, 0.5, 1e-9, 1e9)],
        build=lambda q: stats.expon(scale=1 / q["lam"]),
        r=lambda q: ("exp", f"rate = {q['lam']}"),
        E=r"E(X) = \frac{1}{\lambda}", V=r"Var(X) = \frac{1}{\lambda^2}",
        f=r"f(x) = \lambda e^{-\lambda x},\quad F(x) = 1-e^{-\lambda x}",
        desc="Tiempo entre eventos de un proceso de Poisson (sin memoria).",
    ),
    "Uniforme continua": dict(
        params=[("a", "Mínimo a", float, 0.0, -1e9, 1e9), ("b", "Máximo b", float, 10.0, -1e9, 1e9)],
        build=lambda q: stats.uniform(q["a"], q["b"] - q["a"]),
        r=lambda q: ("unif", f"min = {q['a']}, max = {q['b']}"),
        E=r"E(X) = \frac{a+b}{2}", V=r"Var(X) = \frac{(b-a)^2}{12}",
        f=r"f(x) = \frac{1}{b-a},\quad a\le x\le b",
        desc="Todos los valores del intervalo [a, b] son igualmente probables.",
    ),
    "Gamma": dict(
        params=[("alpha", "Forma α", float, 2.0, 1e-9, 1e6), ("beta", "Escala β", float, 2.0, 1e-9, 1e9)],
        build=lambda q: stats.gamma(q["alpha"], scale=q["beta"]),
        r=lambda q: ("gamma", f"shape = {q['alpha']}, scale = {q['beta']}"),
        E=r"E(X) = \alpha\beta", V=r"Var(X) = \alpha\beta^2",
        f=r"f(x) = \frac{x^{\alpha-1}e^{-x/\beta}}{\beta^\alpha\Gamma(\alpha)}",
        desc="Tiempo hasta el α-ésimo evento; generaliza la exponencial.",
    ),
    "t de Student": dict(
        params=[("gl", "Grados de libertad", float, 10.0, 1e-9, 1e9)],
        build=lambda q: stats.t(q["gl"]),
        r=lambda q: ("t", f"df = {q['gl']}"),
        E=r"E(T) = 0\;(gl>1)", V=r"Var(T) = \frac{gl}{gl-2}\;(gl>2)",
        f=r"T = \frac{\bar X-\mu}{S/\sqrt{n}} \sim t_{n-1}",
        desc="Se usa para inferencia sobre la media con σ desconocida.",
    ),
    "Chi-cuadrado": dict(
        params=[("gl", "Grados de libertad", float, 5.0, 1e-9, 1e9)],
        build=lambda q: stats.chi2(q["gl"]),
        r=lambda q: ("chisq", f"df = {q['gl']}"),
        E=r"E(X) = gl", V=r"Var(X) = 2\,gl",
        f=r"\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}",
        desc="Se usa para inferencia sobre la varianza y pruebas de independencia.",
    ),
    "F de Fisher": dict(
        params=[("d1", "gl numerador", float, 5.0, 1e-9, 1e9), ("d2", "gl denominador", float, 10.0, 1e-9, 1e9)],
        build=lambda q: stats.f(q["d1"], q["d2"]),
        r=lambda q: ("f", f"df1 = {q['d1']}, df2 = {q['d2']}"),
        E=r"E(F) = \frac{d_2}{d_2-2}", V=r"Var(F)=\frac{2d_2^2(d_1+d_2-2)}{d_1(d_2-2)^2(d_2-4)}",
        f=r"F = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}",
        desc="Cociente de varianzas; se usa en ANOVA y regresión.",
    ),
}


def _leer_params(spec, nombre):
    q = {}
    cols = st.columns(len(spec["params"]))
    for col, (k, lab, tipo, dflt, mn, mx) in zip(cols, spec["params"]):
        if tipo is int:
            q[k] = int(col.number_input(lab, int(mn), int(mx), int(dflt), step=1, key=f"dist_{nombre}_{k}"))
        else:
            q[k] = float(col.number_input(lab, float(mn), float(mx), float(dflt), format="%.6g", key=f"dist_{nombre}_{k}"))
    return q


def _validar(nombre, q):
    if nombre == "Hipergeométrica" and (q["K"] > q["N"] or q["n"] > q["N"]):
        return "K y n no pueden ser mayores que N."
    if nombre.startswith("Uniforme") and q["b"] <= q["a"]:
        return "b debe ser mayor que a."
    return None


def _demostracion_puntual(nombre, q, k, valor):
    if nombre == "Binomial":
        n, p = q["n"], q["p"]
        return rf"P(X={k}) = \binom{{{n}}}{{{k}}}({fmt_num(p)})^{{{k}}}(1-{fmt_num(p)})^{{{n - k}}} = {math.comb(n, k)}\cdot{p ** k:.6g}\cdot{(1 - p) ** (n - k):.6g} = {valor:.6f}"
    if nombre == "Poisson":
        lam = q["lam"]
        return rf"P(X={k}) = \frac{{e^{{-{fmt_num(lam)}}}({fmt_num(lam)})^{{{k}}}}}{{{k}!}} = \frac{{{math.exp(-lam):.6g}\cdot{lam ** k:.6g}}}{{{math.factorial(k) if k < 25 else f'{k}!'}}} = {valor:.6f}"
    if nombre == "Hipergeométrica":
        N, K, n = q["N"], q["K"], q["n"]
        return rf"P(X={k}) = \frac{{\binom{{{K}}}{{{k}}}\binom{{{N - K}}}{{{n - k}}}}}{{\binom{{{N}}}{{{n}}}}} = \frac{{{math.comb(K, k) if 0 <= k <= K else 0}\cdot{math.comb(N - K, n - k) if 0 <= n - k <= N - K else 0}}}{{{math.comb(N, n)}}} = {valor:.6f}"
    if nombre == "Geométrica":
        p = q["p"]
        return rf"P(X={k}) = (1-{fmt_num(p)})^{{{k - 1}}}({fmt_num(p)}) = {valor:.6f}"
    if nombre == "Binomial negativa":
        r, p = q["r"], q["p"]
        return rf"P(X={k}) = \binom{{{k - 1}}}{{{r - 1}}}({fmt_num(p)})^{{{r}}}(1-{fmt_num(p)})^{{{k - r}}} = {valor:.6f}"
    if nombre == "Uniforme discreta":
        return rf"P(X={k}) = \frac{{1}}{{{q['b'] - q['a'] + 1}}} = {valor:.6f}"
    return None


def render_distribuciones():
    familia = st.radio("Familia", ["Discretas", "Continuas"], horizontal=True, key="dist_fam")
    catalogo = DISCRETAS if familia == "Discretas" else CONTINUAS
    nombre = st.selectbox("Distribución", list(catalogo), key=f"dist_sel_{familia}")
    spec = catalogo[nombre]
    st.info(spec["desc"])
    q = _leer_params(spec, nombre)
    err = _validar(nombre, q)
    if err:
        st.error(err)
        return
    dist = spec["build"](q)
    discreta = familia == "Discretas"

    c1, c2, c3 = st.columns([2, 2, 3])
    c1.latex(spec["f"])
    c2.latex(spec["E"])
    c2.latex(spec["V"])
    media, var = dist.mean(), dist.var()
    with c3:
        m = st.columns(3)
        m[0].metric("E(X)", fmt_num(media) if np.isfinite(media) else "∞")
        m[1].metric("Var(X)", fmt_num(var) if np.isfinite(var) else "∞")
        m[2].metric("σ", fmt_num(math.sqrt(var)) if np.isfinite(var) else "∞")

    if discreta:
        tipos = ["P(X = k)", "P(X ≤ k)", "P(X < k)", "P(X ≥ k)", "P(X > k)", "P(a ≤ X ≤ b)", "Percentil: menor k con P(X ≤ k) ≥ p"]
    else:
        tipos = ["P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)", "f(x) densidad", "Percentil / valor crítico: x con P(X ≤ x) = p",
                 "Valor crítico de dos colas (±x con área central p)"]
    tipo = st.selectbox("¿Qué quieres calcular?", tipos, key=f"dist_tipo_{familia}")

    lo, hi = dist.ppf(0.0005), dist.ppf(0.9995)
    if discreta:
        lo, hi = int(max(lo, dist.support()[0])), int(hi)
    defecto = media if np.isfinite(media) else dist.median()
    region = None
    pasos = []
    rn = spec["r"](q) if spec["r"] else None

    if tipo.startswith("P(a"):
        c1, c2 = st.columns(2)
        if discreta:
            a = int(c1.number_input("a", value=int(math.floor(defecto)) - 1, step=1, key=f"dist_a_{nombre}"))
            b = int(c2.number_input("b", value=int(math.floor(defecto)) + 1, step=1, key=f"dist_b_{nombre}"))
            res = float(dist.cdf(b) - dist.cdf(a - 1)) if b >= a else 0.0
            pasos.append(rf"P({a}\le X\le {b}) = F({b}) - F({a - 1}) = {dist.cdf(b):.6f} - {dist.cdf(a - 1):.6f} = {res:.6f}")
            region = (a, b)
            r_txt = f"p{rn[0]}({b - rn[2]}, {rn[1]}) - p{rn[0]}({a - 1 - rn[2]}, {rn[1]})" if rn else ""
        else:
            a = c1.number_input("a", value=float(defecto - (math.sqrt(var) if np.isfinite(var) else 1)), key=f"dist_a_{nombre}")
            b = c2.number_input("b", value=float(defecto + (math.sqrt(var) if np.isfinite(var) else 1)), key=f"dist_b_{nombre}")
            res = float(dist.cdf(b) - dist.cdf(a)) if b >= a else 0.0
            pasos.append(rf"P({fmt_num(a)}\le X\le {fmt_num(b)}) = F({fmt_num(b)}) - F({fmt_num(a)}) = {dist.cdf(b):.6f} - {dist.cdf(a):.6f} = {res:.6f}")
            region = (a, b)
            r_txt = f"p{rn[0]}({fmt_num(b)}, {rn[1]}) - p{rn[0]}({fmt_num(a)}, {rn[1]})"
        etiqueta = f"P({fmt_num(a)} ≤ X ≤ {fmt_num(b)})"
    elif tipo.startswith("Percentil") or tipo.startswith("Valor crítico"):
        p = st.number_input("p (probabilidad acumulada / área)", 0.0001, 0.9999, 0.95, format="%.4f", key=f"dist_p_{nombre}")
        if tipo.startswith("Valor crítico de dos"):
            res = float(dist.ppf(0.5 + p / 2))
            etiqueta = f"x tal que P(−x ≤ X ≤ x) = {p}"
            region = (float(dist.ppf(0.5 - p / 2)), res)
            r_txt = f"q{rn[0]}({0.5 + p / 2}, {rn[1]})"
        else:
            res = float(dist.ppf(p))
            etiqueta = f"x tal que P(X ≤ x) {'≥' if discreta else '='} {p}"
            region = (lo if discreta else -np.inf, res)
            r_txt = f"q{rn[0]}({p}, {rn[1]})" + (f" + {rn[2]}" if discreta and rn and rn[2] else "") if rn else ""
        pasos.append(rf"F^{{-1}}({p}) = {fmt_num(res)}")
    else:
        if discreta:
            v = int(st.number_input("k", value=int(math.floor(defecto)), step=1, key=f"dist_k_{nombre}"))
        else:
            v = st.number_input("x", value=float(defecto), key=f"dist_x_{nombre}")
        if tipo == "P(X = k)":
            res = float(dist.pmf(v)); region = (v, v)
            d = _demostracion_puntual(nombre, q, v, res)
            if d:
                pasos.append(d)
            r_txt = f"d{rn[0]}({v - rn[2]}, {rn[1]})" if rn else ""
        elif tipo in ("P(X ≤ k)", "P(X ≤ x)"):
            res = float(dist.cdf(v)); region = (lo if discreta else -np.inf, v)
            r_txt = f"p{rn[0]}({(v - rn[2]) if discreta else fmt_num(v)}, {rn[1]})" if rn else ""
        elif tipo == "P(X < k)":
            res = float(dist.cdf(v - 1)); region = (lo, v - 1)
            pasos.append(rf"P(X<{v}) = P(X\le {v - 1}) = {res:.6f}")
            r_txt = f"p{rn[0]}({v - 1 - rn[2]}, {rn[1]})" if rn else ""
        elif tipo in ("P(X ≥ k)", "P(X ≥ x)"):
            res = float(dist.sf(v - 1)) if discreta else float(dist.sf(v)); region = (v, np.inf)
            pasos.append((rf"P(X\ge {v}) = 1 - P(X\le {v - 1}) = 1 - {dist.cdf(v - 1):.6f} = {res:.6f}") if discreta
                         else (rf"P(X\ge {fmt_num(v)}) = 1 - F({fmt_num(v)}) = 1 - {dist.cdf(v):.6f} = {res:.6f}"))
            r_txt = (f"1 - p{rn[0]}({v - 1 - rn[2]}, {rn[1]})" if discreta else f"1 - p{rn[0]}({fmt_num(v)}, {rn[1]})") if rn else ""
        elif tipo == "P(X > k)":
            res = float(dist.sf(v)); region = (v + 1, np.inf)
            pasos.append(rf"P(X>{v}) = 1 - P(X\le {v}) = 1 - {dist.cdf(v):.6f} = {res:.6f}")
            r_txt = f"1 - p{rn[0]}({v - rn[2]}, {rn[1]})" if rn else ""
        else:  # densidad
            res = float(dist.pdf(v)); region = None
            r_txt = f"d{rn[0]}({fmt_num(v)}, {rn[1]})"
            st.caption("Recuerda: f(x) es una densidad, NO una probabilidad. En una continua P(X = x) = 0.")
        etiqueta = tipo.replace("k", fmt_num(v)).replace("x)", f"{fmt_num(v)})") if "densidad" not in tipo else f"f({fmt_num(v)})"
        if nombre == "Normal" and not discreta:
            z = (v - q["mu"]) / q["sigma"]
            pasos.insert(0, rf"z = \frac{{x-\mu}}{{\sigma}} = \frac{{{fmt_num(v)} - {fmt_num(q['mu'])}}}{{{fmt_num(q['sigma'])}}} = {z:.4f} \;\Rightarrow\; \Phi({z:.4f}) = {stats.norm.cdf(z):.6f}")
        if nombre == "Exponencial" and tipo in ("P(X ≤ x)", "P(X ≥ x)"):
            pasos.append(rf"F({fmt_num(v)}) = 1 - e^{{-{fmt_num(q['lam'])}\cdot{fmt_num(v)}}} = {dist.cdf(v):.6f}")

    st.markdown(f"<div style='background:linear-gradient(90deg,rgba(56,189,248,.16),rgba(56,189,248,.04));border:1px solid rgba(56,189,248,.25);"
                f"padding:16px;border-radius:14px;margin:8px 0'><div style='color:#93c5fd;font-weight:700'>{etiqueta}</div>"
                f"<div style='font-size:1.5rem;font-weight:800'>{res:.6f}"
                f"{'' if tipo.startswith(('Percentil', 'Valor', 'f(x)')) else f' ({res * 100:.4f}%)'}</div></div>", unsafe_allow_html=True)
    if pasos:
        with st.expander("📐 Demostración", expanded=True):
            for p_ in pasos:
                st.latex(p_)
            if discreta and region and tipo != "P(X = k)" and np.isfinite(region[1]) and region[1] - region[0] <= 8:
                ks = list(range(int(max(region[0], lo)), int(region[1]) + 1))
                st.latex(" + ".join(f"P(X={k})" for k in ks) + " = " + " + ".join(f"{dist.pmf(k):.5f}" for k in ks)
                         + f" = {sum(dist.pmf(k) for k in ks):.6f}")

    # Gráfico
    if discreta:
        ks = np.arange(lo, hi + 1)
        pm = dist.pmf(ks)
        dentro = np.zeros_like(ks, bool) if region is None else (ks >= region[0]) & (ks <= region[1])
        fig = go.Figure(go.Bar(x=ks, y=pm, marker_color=[COLOR_A if d else COLOR_PRIMARIO for d in dentro],
                               hovertemplate="P(X=%{x}) = %{y:.6f}<extra></extra>"))
        fig.update_layout(title=f"{nombre}: función de probabilidad", xaxis_title="k", yaxis_title="P(X = k)", bargap=0.15)
    else:
        xs = np.linspace(lo, hi, 600)
        ys = dist.pdf(xs)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line_color=COLOR_PRIMARIO, name="f(x)"))
        if region is not None:
            if tipo.startswith("Valor crítico de dos"):
                for zona in [(xs <= region[0]), (xs >= region[1])]:
                    fig.add_scatter(x=xs[zona], y=ys[zona], fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.35)", showlegend=False)
            else:
                zona = (xs >= region[0]) & (xs <= region[1])
                fig.add_scatter(x=xs[zona], y=ys[zona], fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.35)", name="Área")
        elif "densidad" in tipo:
            fig.add_vline(x=v, line_dash="dash", line_color=COLOR_A)
        fig.update_layout(title=f"{nombre}: función de densidad", xaxis_title="x", yaxis_title="f(x)", showlegend=False)
    if np.isfinite(media):
        fig.add_vline(x=media, line_dash="dot", line_color=COLOR_OK, annotation_text="E(X)")
    mostrar_plotly(fig, key=f"dist_fig_{nombre}")

    if discreta:
        with st.expander("📋 Tabla completa P(X = k) y F(k)"):
            ks = np.arange(lo, hi + 1)
            tabla = pd.DataFrame({"k": ks, "P(X = k)": dist.pmf(ks), "F(k) = P(X ≤ k)": dist.cdf(ks), "P(X > k)": dist.sf(ks)})
            st.dataframe(tabla.style.format({"P(X = k)": "{:.6f}", "F(k) = P(X ≤ k)": "{:.6f}", "P(X > k)": "{:.6f}"}), hide_index=True)
            st.download_button("⬇️ CSV", tabla.to_csv(index=False).encode("utf-8-sig"), f"tabla_{nombre}.csv", key=f"dist_csv_{nombre}")

    if nombre == "Binomial":
        _aproximaciones_binomial(q, tipo, region)

    with st.expander("🧪 Simulación (comparar con la teoría)"):
        c1, c2 = st.columns(2)
        N = c1.slider("Tamaño de la simulación", 100, 100000, 5000, step=100, key=f"dist_simN_{nombre}")
        seed = c2.number_input("Semilla", 0, 10 ** 6, 123, key=f"dist_seed_{nombre}")
        muestra = dist.rvs(size=N, random_state=int(seed))
        m = st.columns(4)
        m[0].metric("Media simulada", fmt_num(np.mean(muestra)))
        m[1].metric("Media teórica", fmt_num(media) if np.isfinite(media) else "∞")
        m[2].metric("Var simulada", fmt_num(np.var(muestra, ddof=1)))
        m[3].metric("Var teórica", fmt_num(var) if np.isfinite(var) else "∞")
        fig = go.Figure()
        if discreta:
            vals, cnt = np.unique(muestra, return_counts=True)
            fig.add_bar(x=vals, y=cnt / N, name="Simulada", marker_color=COLOR_OK, opacity=0.7)
            ks = np.arange(lo, hi + 1)
            fig.add_scatter(x=ks, y=dist.pmf(ks), mode="markers", name="Teórica", marker=dict(color=COLOR_A, size=8))
        else:
            fig.add_histogram(x=muestra, histnorm="probability density", name="Simulada", marker_color=COLOR_OK, opacity=0.7, nbinsx=60)
            xs = np.linspace(lo, hi, 400)
            fig.add_scatter(x=xs, y=dist.pdf(xs), mode="lines", name="Teórica", line_color=COLOR_A)
        fig.update_layout(title="Simulación vs distribución teórica", barmode="overlay")
        mostrar_plotly(fig, key=f"dist_sim_{nombre}")
        if rn:
            st.code(f"set.seed({int(seed)})\nx <- r{rn[0]}({N}, {rn[1]})" + (f" + {rn[2]}" if discreta and rn[2] else "")
                    + "\nmean(x); var(x)\nhist(x, freq = FALSE)", language="r")

    if rn and r_txt:
        nota = ""
        if discreta and rn[2]:
            nota = f"\n# Ojo: en R, {rn[0]} cuenta FRACASOS antes del éxito; por eso se resta {rn[2]} a k."
        codigo_r(f"# {etiqueta}\n{r_txt}{nota}")
    elif nombre == "Uniforme discreta":
        codigo_r(f"x <- {q['a']}:{q['b']}\nmean(x == k)    # P(X = k)\nmean(x <= k)    # P(X <= k)")
    reporte(f"Para una distribución {nombre} con parámetros {', '.join(f'{k} = {v}' for k, v in q.items())}, "
            f"se obtiene {etiqueta} = {res:.6f}. La media de la distribución es {fmt_num(media) if np.isfinite(media) else '∞'} "
            f"y su desviación estándar {fmt_num(math.sqrt(var)) if np.isfinite(var) else '∞'}.", f"dist_{nombre}", "Distribuciones")


def _aproximaciones_binomial(q, tipo, region):
    n, p = q["n"], q["p"]
    with st.expander("≈ Aproximaciones: Normal y Poisson a la Binomial"):
        mu, sd = n * p, math.sqrt(n * p * (1 - p))
        ok_normal = n * p >= 5 and n * (1 - p) >= 5
        ok_poisson = n >= 20 and p <= 0.05
        st.markdown(f"- Normal: np = {n * p:.2f}, n(1−p) = {n * (1 - p):.2f} → {'✅ adecuada' if ok_normal else '⚠️ no recomendada (se requiere ≥ 5)'}")
        st.markdown(f"- Poisson: n = {n}, p = {p} → {'✅ adecuada' if ok_poisson else '⚠️ no recomendada (n ≥ 20 y p ≤ 0.05)'}")
        if region is None or tipo.startswith("Percentil"):
            return
        a, b = region
        a = max(a, 0)
        b = min(b, n)
        exacta = stats.binom(n, p).cdf(b) - stats.binom(n, p).cdf(a - 1)
        normal = stats.norm(mu, sd).cdf(b + 0.5) - stats.norm(mu, sd).cdf(a - 0.5) if sd > 0 else float("nan")
        poisson = stats.poisson(mu).cdf(b) - stats.poisson(mu).cdf(a - 1)
        st.latex(rf"P({int(a)}\le X\le {int(b)}) \approx P\left(\frac{{{int(a)}-0.5-{mu:.3f}}}{{{sd:.3f}}} \le Z \le \frac{{{int(b)}+0.5-{mu:.3f}}}{{{sd:.3f}}}\right)\quad\text{{(corrección por continuidad)}}")
        st.dataframe(pd.DataFrame({"Método": ["Binomial exacta", "Aprox. Normal (con corrección)", "Aprox. Poisson (λ = np)"],
                                   "Probabilidad": [exacta, normal, poisson],
                                   "Error absoluto": [0, abs(normal - exacta), abs(poisson - exacta)]}).style.format(precision=6), hide_index=True)


# ------------------------------------------------------------------
# Teorema del límite central
# ------------------------------------------------------------------

def render_tlc():
    st.markdown("El **Teorema del Límite Central** dice que la media muestral X̄ se distribuye aproximadamente "
                "Normal(μ, σ/√n) para n grande, sin importar la forma de la población.")
    c1, c2, c3, c4 = st.columns(4)
    pob = c1.selectbox("Población", ["Exponencial (sesgada)", "Uniforme", "Bernoulli p=0.2", "Poisson λ=2", "Bimodal"], key="tlc_pob")
    n = c2.slider("Tamaño de cada muestra n", 1, 200, 30, key="tlc_n")
    M = c3.slider("Número de muestras", 100, 20000, 3000, step=100, key="tlc_M")
    seed = c4.number_input("Semilla", 0, 10 ** 6, 1, key="tlc_seed")
    rng = np.random.default_rng(int(seed))
    if pob.startswith("Exponencial"):
        gen, mu, sigma = (lambda s: rng.exponential(1.0, s)), 1.0, 1.0
    elif pob == "Uniforme":
        gen, mu, sigma = (lambda s: rng.uniform(0, 1, s)), 0.5, math.sqrt(1 / 12)
    elif pob.startswith("Bernoulli"):
        gen, mu, sigma = (lambda s: rng.binomial(1, 0.2, s)), 0.2, math.sqrt(0.16)
    elif pob.startswith("Poisson"):
        gen, mu, sigma = (lambda s: rng.poisson(2, s)), 2.0, math.sqrt(2)
    else:
        def gen(s):
            comp = rng.random(s) < 0.5
            return np.where(comp, rng.normal(-2, 0.7, s), rng.normal(2, 0.7, s))
        mu, sigma = 0.0, math.sqrt(0.49 + 4)
    datos = gen((M, n))
    medias = datos.mean(axis=1)
    se = sigma / math.sqrt(n)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Histogram(x=gen(20000), histnorm="probability density", marker_color=COLOR_GRIS, nbinsx=60))
        fig.update_layout(title="Población original", showlegend=False)
        mostrar_plotly(fig, key="tlc_pob_fig")
    with c2:
        fig = go.Figure(go.Histogram(x=medias, histnorm="probability density", marker_color=COLOR_PRIMARIO, nbinsx=60, name="X̄ simuladas"))
        xs = np.linspace(medias.min(), medias.max(), 300)
        fig.add_scatter(x=xs, y=stats.norm.pdf(xs, mu, se), mode="lines", line_color=COLOR_A, name="Normal(μ, σ/√n)")
        fig.update_layout(title=f"Distribución de X̄ con n = {n}", legend=dict(orientation="h"))
        mostrar_plotly(fig, key="tlc_med_fig")
    m = st.columns(4)
    m[0].metric("μ poblacional", fmt_num(mu))
    m[1].metric("Media de las X̄", fmt_num(medias.mean()))
    m[2].metric("σ/√n teórico", fmt_num(se))
    m[3].metric("Desv. de las X̄", fmt_num(medias.std(ddof=1)))
    st.latex(rf"\mu_{{\bar X}} = \mu = {fmt_num(mu)}, \qquad \sigma_{{\bar X}} = \frac{{\sigma}}{{\sqrt{{n}}}} = \frac{{{fmt_num(sigma)}}}{{\sqrt{{{n}}}}} = {fmt_num(se)}")
    codigo_r(f"set.seed({int(seed)})\nmedias <- replicate({M}, mean(rexp({n}, rate = 1)))  # cambia rexp por tu población\n"
             f"hist(medias, freq = FALSE)\ncurve(dnorm(x, 1, 1/sqrt({n})), add = TRUE, col = 'red')")
