"""CA4 · Complementos de inferencia: gráficos de región de rechazo, IC adicionales y prueba para la varianza."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from .comunes import COLOR_A, COLOR_OK, COLOR_PRIMARIO, codigo_r, fmt_num, mostrar_plotly, reporte

TEMAS_IC_EXTRA = ["Varianza y desviación (χ²)", "Diferencia de medias", "Diferencia de proporciones",
                  "Tamaño de muestra", "Simulación: ¿qué significa la confianza?"]


def grafico_region_rechazo(estad, tipo, alpha, dist="z", gl=None, key="rr", gl2=None):
    """Dibuja la distribución bajo H0, sombrea la región de rechazo y marca el estadístico."""
    try:
        if dist == "z":
            D, nombre = stats.norm(), "z"
        elif dist == "t":
            D, nombre = stats.t(gl), f"t (gl = {fmt_num(gl, 2)})"
        else:
            D, nombre = stats.chi2(gl), f"χ² (gl = {gl})"
        lo, hi = D.ppf(0.0005), D.ppf(0.9995)
        if np.isfinite(estad):
            lo, hi = min(lo, estad - 0.5), max(hi, estad + 0.5)
        xs = np.linspace(lo, hi, 600)
        ys = D.pdf(xs)
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", line_color=COLOR_PRIMARIO, name="Distribución bajo H₀"))
        if tipo == "Bilateral":
            cortes = [(xs <= D.ppf(alpha / 2)), (xs >= D.ppf(1 - alpha / 2))]
            crit = [D.ppf(alpha / 2), D.ppf(1 - alpha / 2)]
        elif tipo == "Cola izquierda":
            cortes, crit = [(xs <= D.ppf(alpha))], [D.ppf(alpha)]
        else:
            cortes, crit = [(xs >= D.ppf(1 - alpha))], [D.ppf(1 - alpha)]
        for i, zona in enumerate(cortes):
            fig.add_scatter(x=xs[zona], y=ys[zona], fill="tozeroy", mode="none", fillcolor="rgba(239,68,68,0.4)",
                            name="Región de rechazo", showlegend=i == 0)
        for c in crit:
            fig.add_vline(x=c, line_dash="dot", line_color=COLOR_A, annotation_text=f"crítico {c:.3f}")
        if np.isfinite(estad):
            fig.add_vline(x=estad, line_width=3, line_color=COLOR_OK, annotation_text=f"calculado {estad:.3f}", annotation_position="top left")
        fig.update_layout(title=f"Región de rechazo · {nombre} · α = {alpha}", xaxis_title=nombre, yaxis_title="densidad",
                          height=340, legend=dict(orientation="h"))
        mostrar_plotly(fig, key=key)
    except Exception as e:  # el gráfico nunca debe romper el cálculo
        st.caption(f"(No se pudo dibujar la región de rechazo: {e})")


def grafico_intervalo(li, ls, centro, etiqueta="IC", key="gic"):
    fig = go.Figure()
    fig.add_scatter(x=[li, ls], y=[0, 0], mode="lines+markers", line=dict(color=COLOR_PRIMARIO, width=8), marker=dict(size=14),
                    hovertemplate="%{x:.4f}<extra></extra>")
    fig.add_scatter(x=[centro], y=[0], mode="markers+text", marker=dict(size=16, color=COLOR_A), text=[f"{centro:.4f}"], textposition="top center")
    for v in (li, ls):
        fig.add_annotation(x=v, y=0, text=f"{v:.4f}", showarrow=False, yshift=-22)
    fig.update_layout(title=etiqueta, height=180, showlegend=False, yaxis=dict(visible=False), margin=dict(t=40, b=20))
    mostrar_plotly(fig, key=key)


def _nc(key):
    txt = st.text_input("Nivel de confianza", "95%", key=key)
    try:
        v = float(txt.replace("%", "").replace(",", "."))
        v = v / 100 if v > 1 else v
        if not 0 < v < 1:
            raise ValueError
        return v
    except Exception:
        st.error("Nivel de confianza no válido; se usa 95%.")
        return 0.95


def render_ic_extra(tema):
    if tema.startswith("Varianza"):
        c1, c2, c3 = st.columns(3)
        with c1:
            nc = _nc("icv_nc")
        n = int(c2.number_input("n", 2, 100000, 20, key="icv_n"))
        modo = c3.radio("Dato muestral", ["s (desviación)", "s² (varianza)"], key="icv_modo")
        val = st.number_input("Valor", 0.0001, value=4.0, key="icv_val")
        s2 = val ** 2 if modo.startswith("s (") else val
        a = 1 - nc
        gl = n - 1
        chi_d, chi_i = stats.chi2.ppf(1 - a / 2, gl), stats.chi2.ppf(a / 2, gl)
        li, ls = gl * s2 / chi_d, gl * s2 / chi_i
        st.latex(r"IC_{\sigma^2} = \left(\frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}},\;\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}}\right)")
        st.latex(rf"= \left(\frac{{{gl}\cdot{fmt_num(s2)}}}{{{chi_d:.4f}}},\;\frac{{{gl}\cdot{fmt_num(s2)}}}{{{chi_i:.4f}}}\right) = ({li:.4f},\;{ls:.4f})")
        st.success(f"IC {nc * 100:.1f}% para σ²: ({li:.4f}, {ls:.4f})   ·   para σ: ({math.sqrt(li):.4f}, {math.sqrt(ls):.4f})")
        grafico_intervalo(math.sqrt(li), math.sqrt(ls), math.sqrt(s2), "Intervalo para σ (no es simétrico)", "icv_g")
        codigo_r(f"n <- {n}; s2 <- {fmt_num(s2)}; a <- {1 - nc:.4f}\nc((n-1)*s2/qchisq(1-a/2, n-1), (n-1)*s2/qchisq(a/2, n-1))")
        reporte(f"Con un {nc * 100:.1f}% de confianza, la varianza poblacional está entre {li:.4f} y {ls:.4f}, es decir, la "
                f"desviación estándar poblacional está entre {math.sqrt(li):.4f} y {math.sqrt(ls):.4f}.", "icv", "IC varianza")

    elif tema == "Diferencia de medias":
        metodo = st.selectbox("Método", ["Z (σ₁, σ₂ conocidas o n grandes)", "t con varianzas iguales (pooled)", "t de Welch (varianzas distintas)"], key="icdm_m")
        nc = _nc("icdm_nc")
        c1, c2 = st.columns(2)
        x1 = c1.number_input("x̄₁", value=75.0, key="icdm_x1"); s1 = c1.number_input("s₁ (o σ₁)", 0.0001, value=8.0, key="icdm_s1"); n1 = int(c1.number_input("n₁", 2, value=40, key="icdm_n1"))
        x2 = c2.number_input("x̄₂", value=70.0, key="icdm_x2"); s2 = c2.number_input("s₂ (o σ₂)", 0.0001, value=10.0, key="icdm_s2"); n2 = int(c2.number_input("n₂", 2, value=35, key="icdm_n2"))
        a = 1 - nc
        d = x1 - x2
        if metodo.startswith("Z"):
            se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2); crit = stats.norm.ppf(1 - a / 2); gl_txt = "—"
            st.latex(r"(\bar x_1-\bar x_2) \pm z_{\alpha/2}\sqrt{\frac{\sigma_1^2}{n_1}+\frac{\sigma_2^2}{n_2}}")
        elif "pooled" in metodo:
            gl = n1 + n2 - 2
            sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / gl)
            se = sp * math.sqrt(1 / n1 + 1 / n2); crit = stats.t.ppf(1 - a / 2, gl); gl_txt = str(gl)
            st.latex(rf"s_p = {sp:.4f},\qquad (\bar x_1-\bar x_2) \pm t_{{\alpha/2,\,n_1+n_2-2}}\,s_p\sqrt{{\tfrac{{1}}{{n_1}}+\tfrac{{1}}{{n_2}}}}")
        else:
            v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
            gl = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
            se = math.sqrt(v1 + v2); crit = stats.t.ppf(1 - a / 2, gl); gl_txt = f"{gl:.2f}"
            st.latex(r"(\bar x_1-\bar x_2) \pm t_{\alpha/2,\,gl_{Welch}}\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}}")
        li, ls = d - crit * se, d + crit * se
        m = st.columns(4)
        m[0].metric("x̄₁ − x̄₂", fmt_num(d)); m[1].metric("SE", fmt_num(se)); m[2].metric("Valor crítico", fmt_num(crit)); m[3].metric("gl", gl_txt)
        st.success(f"IC {nc * 100:.1f}% para μ₁ − μ₂: ({li:.4f}, {ls:.4f})")
        grafico_intervalo(li, ls, d, "Intervalo para μ₁ − μ₂ (¿contiene al 0?)", "icdm_g")
        contiene = li <= 0 <= ls
        st.info("El intervalo contiene el 0: no hay evidencia de diferencia entre las medias." if contiene
                else "El intervalo NO contiene el 0: hay evidencia de que las medias son diferentes.")
        codigo_r("# Con los datos crudos:\nt.test(grupo1, grupo2, var.equal = " + ("TRUE" if "pooled" in metodo else "FALSE") + f", conf.level = {nc})")
        reporte(f"Con un {nc * 100:.1f}% de confianza, la diferencia entre las medias poblacionales (μ₁ − μ₂) está entre {li:.4f} y {ls:.4f}. "
                + ("Como el intervalo incluye el cero, no se puede afirmar que las medias sean distintas." if contiene
                   else "Como el intervalo no incluye el cero, se concluye que las medias son distintas."), "icdm", "IC diferencia de medias")

    elif tema == "Diferencia de proporciones":
        nc = _nc("icdp_nc")
        c1, c2 = st.columns(2)
        x1 = int(c1.number_input("x₁ éxitos", 0, value=120, key="icdp_x1")); n1 = int(c1.number_input("n₁", 1, value=300, key="icdp_n1"))
        x2 = int(c2.number_input("x₂ éxitos", 0, value=90, key="icdp_x2")); n2 = int(c2.number_input("n₂", 1, value=280, key="icdp_n2"))
        if x1 > n1 or x2 > n2:
            st.error("Los éxitos no pueden superar al tamaño de muestra.")
            return
        p1, p2 = x1 / n1, x2 / n2
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        z = stats.norm.ppf(1 - (1 - nc) / 2)
        li, ls = p1 - p2 - z * se, p1 - p2 + z * se
        st.latex(r"(\hat p_1-\hat p_2) \pm z_{\alpha/2}\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}")
        st.latex(rf"= ({p1:.4f}-{p2:.4f}) \pm {z:.4f}\cdot{se:.4f} = ({li:.4f},\;{ls:.4f})")
        st.success(f"IC {nc * 100:.1f}% para p₁ − p₂: ({li:.4f}, {ls:.4f})  →  ({li * 100:.2f}%, {ls * 100:.2f}%)")
        grafico_intervalo(li, ls, p1 - p2, "Intervalo para p₁ − p₂", "icdp_g")
        codigo_r(f"prop.test(c({x1}, {x2}), c({n1}, {n2}), conf.level = {nc}, correct = FALSE)")
        reporte(f"Con un {nc * 100:.1f}% de confianza, la diferencia entre las proporciones poblacionales está entre "
                f"{li * 100:.2f}% y {ls * 100:.2f}%.", "icdp", "IC diferencia de proporciones")

    elif tema == "Tamaño de muestra":
        obj = st.radio("Para estimar", ["Una media μ", "Una proporción p"], horizontal=True, key="tm_obj")
        nc = _nc("tm_nc")
        z = stats.norm.ppf(1 - (1 - nc) / 2)
        c1, c2, c3 = st.columns(3)
        E = c1.number_input("Margen de error E", 0.000001, value=2.0 if obj.endswith("μ") else 0.05, format="%.4f", key="tm_E")
        if obj.endswith("μ"):
            sigma = c2.number_input("σ (estimada)", 0.0001, value=10.0, key="tm_sigma")
            n0 = (z * sigma / E) ** 2
            st.latex(rf"n = \left(\frac{{z_{{\alpha/2}}\,\sigma}}{{E}}\right)^2 = \left(\frac{{{z:.4f}\cdot{fmt_num(sigma)}}}{{{fmt_num(E)}}}\right)^2 = {n0:.4f}")
        else:
            p = c2.number_input("p estimada (0.5 si no se sabe)", 0.0, 1.0, 0.5, key="tm_p")
            n0 = z ** 2 * p * (1 - p) / E ** 2
            st.latex(rf"n = \frac{{z_{{\alpha/2}}^2\,p(1-p)}}{{E^2}} = \frac{{{z:.4f}^2\cdot{fmt_num(p)}\cdot{fmt_num(1 - p)}}}{{{fmt_num(E)}^2}} = {n0:.4f}")
        N = c3.number_input("Tamaño de población N (0 = infinita)", 0, value=0, key="tm_N")
        n = math.ceil(n0 - 1e-9)
        if N > 0:
            nf = n0 / (1 + (n0 - 1) / N)
            st.latex(rf"n_{{ajustado}} = \frac{{n_0}}{{1 + \frac{{n_0-1}}{{N}}}} = {nf:.4f}")
            n = math.ceil(nf - 1e-9)
        st.success(f"Tamaño de muestra necesario: n = {n} (siempre se redondea hacia arriba)")
        reporte(f"Para estimar {'la media' if obj.endswith('μ') else 'la proporción'} con un margen de error de {fmt_num(E)} y "
                f"{nc * 100:.1f}% de confianza se necesita una muestra de al menos {n} elementos.", "tm", "Tamaño de muestra")

    else:
        st.markdown("Se toman muchas muestras de una población **Normal(μ, σ)** y se calcula un IC para cada una. "
                    "Aproximadamente el nivel de confianza de ellos debería contener a la verdadera μ.")
        c1, c2, c3, c4 = st.columns(4)
        mu = c1.number_input("μ real", value=50.0, key="sim_ic_mu")
        sigma = c2.number_input("σ real", 0.1, value=10.0, key="sim_ic_s")
        n = int(c3.number_input("n", 2, 1000, 25, key="sim_ic_n"))
        M = int(c4.number_input("Número de intervalos", 10, 300, 100, key="sim_ic_M"))
        nc = st.slider("Nivel de confianza", 0.50, 0.99, 0.95, 0.01, key="sim_ic_nc")
        seed = st.number_input("Semilla (cámbiala para otra simulación)", 0, 10 ** 6, 11, key="sim_ic_seed")
        rng = np.random.default_rng(int(seed))
        datos = rng.normal(mu, sigma, (M, n))
        xb = datos.mean(axis=1)
        s = datos.std(axis=1, ddof=1)
        t = stats.t.ppf(1 - (1 - nc) / 2, n - 1)
        li, ls = xb - t * s / math.sqrt(n), xb + t * s / math.sqrt(n)
        ok = (li <= mu) & (mu <= ls)
        fig = go.Figure()
        for i in range(M):
            fig.add_scatter(x=[li[i], ls[i]], y=[i, i], mode="lines", line=dict(color=COLOR_PRIMARIO if ok[i] else COLOR_A, width=2),
                            showlegend=False, hovertemplate=f"IC #{i + 1}: ({li[i]:.2f}, {ls[i]:.2f})<extra></extra>")
        fig.add_vline(x=mu, line_color=COLOR_OK, line_width=2, annotation_text="μ real")
        fig.update_layout(title=f"{ok.sum()} de {M} intervalos ({ok.mean() * 100:.1f}%) contienen a μ", height=max(400, M * 5),
                          yaxis=dict(visible=False), xaxis_title="valor")
        mostrar_plotly(fig, key="sim_ic_fig")
        st.info(f"Cobertura observada: {ok.mean() * 100:.1f}% vs nominal {nc * 100:.0f}%. Los intervalos rojos no contienen a μ: "
                "la confianza es una propiedad del MÉTODO, no de un intervalo individual.")


def render_prueba_varianza():
    c1, c2, c3 = st.columns(3)
    s0 = c1.number_input("σ₀² hipotética (varianza)", 0.0001, value=25.0, key="pv_s0")
    s2 = c2.number_input("s² muestral", 0.0001, value=36.0, key="pv_s2")
    n = int(c3.number_input("n", 2, 100000, 20, key="pv_n"))
    c1, c2 = st.columns(2)
    alpha = c1.number_input("α", 0.001, 0.5, 0.05, key="pv_alpha")
    tipo = c2.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"], key="pv_tipo")
    gl = n - 1
    chi = gl * s2 / s0
    D = stats.chi2(gl)
    if tipo == "Bilateral":
        p = 2 * min(D.cdf(chi), D.sf(chi))
    elif tipo == "Cola izquierda":
        p = D.cdf(chi)
    else:
        p = D.sf(chi)
    st.latex(rf"\chi^2 = \frac{{(n-1)s^2}}{{\sigma_0^2}} = \frac{{{gl}\cdot{fmt_num(s2)}}}{{{fmt_num(s0)}}} = {chi:.4f},\qquad gl = {gl}")
    m = st.columns(3)
    m[0].metric("χ² calculado", f"{chi:.4f}")
    m[1].metric("p-valor", f"{p:.6f}")
    m[2].metric("Decisión", "Rechazar H₀" if p < alpha else "No rechazar H₀")
    grafico_region_rechazo(chi, tipo, alpha, "chi2", gl, key="pv_rr")
    h1 = {"Bilateral": "la varianza poblacional es diferente de", "Cola izquierda": "la varianza poblacional es menor que",
          "Cola derecha": "la varianza poblacional es mayor que"}[tipo]
    codigo_r(f"chi <- ({n}-1)*{fmt_num(s2)}/{fmt_num(s0)}\n"
             + {"Bilateral": f"2*min(pchisq(chi, {gl}), 1 - pchisq(chi, {gl}))", "Cola izquierda": f"pchisq(chi, {gl})",
                "Cola derecha": f"1 - pchisq(chi, {gl})"}[tipo]
             + "\n# con datos crudos: EnvStats::varTest(x, sigma.squared = σ0²)")
    reporte(f"Con α = {alpha}, el estadístico χ² = {chi:.4f} (gl = {gl}) tiene un p-valor de {p:.4f}, por lo que "
            f"{'se rechaza' if p < alpha else 'no se rechaza'} H₀. {'Existe' if p < alpha else 'No existe'} evidencia suficiente para afirmar "
            f"que {h1} {fmt_num(s0)}.", "pv", "Prueba de varianza")
