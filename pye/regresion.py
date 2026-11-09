"""CA5 · Regresión lineal simple y múltiple."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from .comunes import COLOR_A, COLOR_ACENTO, COLOR_OK, COLOR_PRIMARIO, codigo_r, fmt_num, mostrar_plotly, parse_lista, reporte

EJ_X = "1.2, 2.0, 2.9, 3.8, 4.5, 5.1, 6.3, 7.0, 7.8, 9.1"
EJ_Y = "3.1, 4.8, 6.2, 7.9, 9.4, 10.1, 12.8, 13.9, 15.2, 18.0"


def render_regresion(lector):
    tema = st.radio("Modelo", ["Regresión lineal simple", "Regresión lineal múltiple"], horizontal=True, key="reg_tema")
    if tema.endswith("simple"):
        _simple(lector)
    else:
        _multiple(lector)


def _datos_simple(lector):
    fuente = st.radio("Fuente", ["Datos manuales", "Archivo CSV o Excel"], horizontal=True, key="rs_fuente")
    if fuente == "Datos manuales":
        c1, c2 = st.columns(2)
        nx = c1.text_input("Nombre de X (independiente)", "Horas de estudio", key="rs_nx")
        tx_ = c1.text_area("Valores de X", EJ_X, key="rs_x")
        ny = c2.text_input("Nombre de Y (dependiente)", "Nota", key="rs_ny")
        ty = c2.text_area("Valores de Y", EJ_Y, key="rs_y")
        try:
            x, y = np.array(parse_lista(tx_)), np.array(parse_lista(ty))
        except Exception:
            st.error("Hay valores no numéricos.")
            return None
        return x, y, nx, ny
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="rs_file")
    if archivo is None:
        return None
    df = lector(archivo.getvalue(), archivo.name.lower())
    num = df.select_dtypes("number").columns.tolist()
    if len(num) < 2:
        st.error("Se necesitan al menos 2 columnas numéricas.")
        return None
    c1, c2 = st.columns(2)
    nx = c1.selectbox("X (independiente)", num, key="rs_cx")
    ny = c2.selectbox("Y (dependiente)", [c for c in num if c != nx], key="rs_cy")
    d = df[[nx, ny]].dropna()
    return d[nx].to_numpy(float), d[ny].to_numpy(float), nx, ny


def _simple(lector):
    res = _datos_simple(lector)
    if res is None:
        return
    x, y, nx, ny = res
    if len(x) != len(y):
        st.error(f"X tiene {len(x)} datos y Y tiene {len(y)}; deben tener la misma cantidad.")
        return
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        st.error("Se necesitan al menos 3 datos y X no puede ser constante.")
        return
    alpha = st.select_slider("Nivel de significancia α", [0.01, 0.05, 0.10], 0.05, key="rs_alpha")
    xb, yb = x.mean(), y.mean()
    Sxx = np.sum((x - xb) ** 2)
    Syy = np.sum((y - yb) ** 2)
    Sxy = np.sum((x - xb) * (y - yb))
    b1 = Sxy / Sxx
    b0 = yb - b1 * xb
    yhat = b0 + b1 * x
    e = y - yhat
    SSE = np.sum(e ** 2)
    SST = Syy
    SSR = SST - SSE
    R2 = SSR / SST if SST else float("nan")
    r = Sxy / math.sqrt(Sxx * Syy) if Syy else float("nan")
    gl = n - 2
    MSE = SSE / gl
    s = math.sqrt(MSE)
    se_b1 = s / math.sqrt(Sxx)
    se_b0 = s * math.sqrt(1 / n + xb ** 2 / Sxx)
    tc = stats.t.ppf(1 - alpha / 2, gl)
    t_b1, t_b0 = b1 / se_b1 if se_b1 else float("inf"), b0 / se_b0 if se_b0 else float("inf")
    p_b1, p_b0 = 2 * stats.t.sf(abs(t_b1), gl), 2 * stats.t.sf(abs(t_b0), gl)
    F = SSR / MSE if MSE else float("inf")
    pF = stats.f.sf(F, 1, gl)

    st.markdown("### 📈 Modelo estimado")
    signo = "+" if b1 >= 0 else "-"
    st.latex(rf"\hat{{y}} = {fmt_num(b0)} {signo} {fmt_num(abs(b1))}\,x")
    c = st.columns(5)
    c[0].metric("β₀ (intercepto)", fmt_num(b0))
    c[1].metric("β₁ (pendiente)", fmt_num(b1))
    c[2].metric("r", fmt_num(r))
    c[3].metric("R²", f"{R2 * 100:.2f}%")
    c[4].metric("Error estándar s", fmt_num(s))

    t1, t2, t3, t4, t5 = st.tabs(["📊 Gráfico", "📐 Demostración", "🧪 Inferencia y ANOVA", "🔍 Residuos", "🔮 Predicción"])
    with t1:
        c1, c2 = st.columns(2)
        banda_m = c1.checkbox("Banda de confianza para la media", True, key="rs_bm")
        banda_p = c2.checkbox("Banda de predicción individual", False, key="rs_bp")
        xs = np.linspace(x.min(), x.max(), 200)
        ys = b0 + b1 * xs
        fig = go.Figure()
        if banda_p:
            hp = tc * s * np.sqrt(1 + 1 / n + (xs - xb) ** 2 / Sxx)
            fig.add_scatter(x=np.concatenate([xs, xs[::-1]]), y=np.concatenate([ys + hp, (ys - hp)[::-1]]), fill="toself",
                            fillcolor="rgba(245,158,11,0.15)", line_width=0, name=f"Predicción {100 * (1 - alpha):.0f}%")
        if banda_m:
            hm = tc * s * np.sqrt(1 / n + (xs - xb) ** 2 / Sxx)
            fig.add_scatter(x=np.concatenate([xs, xs[::-1]]), y=np.concatenate([ys + hm, (ys - hm)[::-1]]), fill="toself",
                            fillcolor="rgba(56,189,248,0.25)", line_width=0, name=f"Confianza {100 * (1 - alpha):.0f}%")
        fig.add_scatter(x=x, y=y, mode="markers", name="Datos", marker=dict(color=COLOR_PRIMARIO, size=9),
                        hovertemplate=f"{nx} = %{{x}}<br>{ny} = %{{y}}<extra></extra>")
        fig.add_scatter(x=xs, y=ys, mode="lines", name="Recta ajustada", line=dict(color=COLOR_A, width=3))
        fig.update_layout(title=f"{ny} vs {nx}", xaxis_title=nx, yaxis_title=ny, legend=dict(orientation="h"))
        mostrar_plotly(fig, key="rs_fig")
    with t2:
        tabla = pd.DataFrame({"x": x, "y": y, "x − x̄": x - xb, "y − ȳ": y - yb, "(x−x̄)²": (x - xb) ** 2,
                              "(x−x̄)(y−ȳ)": (x - xb) * (y - yb), "ŷ": yhat, "e = y − ŷ": e, "e²": e ** 2})
        st.dataframe(tabla.style.format(precision=4), hide_index=True)
        st.latex(rf"\bar{{x}} = {fmt_num(xb)},\quad \bar{{y}} = {fmt_num(yb)}")
        st.latex(rf"S_{{xx}} = \sum (x-\bar{{x}})^2 = {fmt_num(Sxx)},\quad S_{{xy}} = \sum (x-\bar{{x}})(y-\bar{{y}}) = {fmt_num(Sxy)},\quad S_{{yy}} = {fmt_num(Syy)}")
        st.latex(rf"\hat\beta_1 = \frac{{S_{{xy}}}}{{S_{{xx}}}} = \frac{{{fmt_num(Sxy)}}}{{{fmt_num(Sxx)}}} = {fmt_num(b1)}")
        st.latex(rf"\hat\beta_0 = \bar{{y}} - \hat\beta_1\bar{{x}} = {fmt_num(yb)} - ({fmt_num(b1)})({fmt_num(xb)}) = {fmt_num(b0)}")
        st.latex(rf"r = \frac{{S_{{xy}}}}{{\sqrt{{S_{{xx}}S_{{yy}}}}}} = {fmt_num(r)},\qquad R^2 = \frac{{SSR}}{{SST}} = \frac{{{fmt_num(SSR)}}}{{{fmt_num(SST)}}} = {fmt_num(R2)}")
        st.latex(rf"s = \sqrt{{\frac{{SSE}}{{n-2}}}} = \sqrt{{\frac{{{fmt_num(SSE)}}}{{{gl}}}}} = {fmt_num(s)}")
    with t3:
        coef = pd.DataFrame({
            "Coeficiente": ["β₀ (intercepto)", f"β₁ ({nx})"], "Estimación": [b0, b1], "Error estándar": [se_b0, se_b1],
            "t": [t_b0, t_b1], "p-valor": [p_b0, p_b1],
            f"IC {100 * (1 - alpha):.0f}% inf": [b0 - tc * se_b0, b1 - tc * se_b1],
            f"IC {100 * (1 - alpha):.0f}% sup": [b0 + tc * se_b0, b1 + tc * se_b1]})
        st.dataframe(coef.style.format(precision=5), hide_index=True)
        st.latex(rf"H_0:\beta_1 = 0 \quad vs \quad H_1:\beta_1\neq 0,\qquad t = \frac{{\hat\beta_1}}{{SE(\hat\beta_1)}} = \frac{{{fmt_num(b1)}}}{{{fmt_num(se_b1)}}} = {fmt_num(t_b1)},\quad t_{{{alpha / 2},{gl}}} = {fmt_num(tc)}")
        if p_b1 < alpha:
            st.success(f"p-valor = {p_b1:.6f} < α = {alpha}: se rechaza H₀. {nx} es un predictor significativo de {ny}.")
        else:
            st.warning(f"p-valor = {p_b1:.6f} ≥ α = {alpha}: no se rechaza H₀. No hay evidencia de relación lineal significativa.")
        st.markdown("**Tabla ANOVA**")
        st.dataframe(pd.DataFrame({"Fuente": ["Regresión", "Error", "Total"], "SC": [SSR, SSE, SST], "gl": [1, gl, n - 1],
                                   "CM": [SSR, MSE, None], "F": [F, None, None], "p-valor": [pF, None, None]})
                     .style.format(precision=5, na_rep=""), hide_index=True)
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=yhat, y=e, mode="markers", marker_color=COLOR_PRIMARIO))
            fig.add_hline(y=0, line_dash="dash", line_color=COLOR_A)
            fig.update_layout(title="Residuos vs valores ajustados", xaxis_title="ŷ", yaxis_title="e")
            mostrar_plotly(fig, key="rs_res")
        with c2:
            osm, osr = stats.probplot(e, dist="norm")[0]
            fig = go.Figure(go.Scatter(x=osm, y=osr, mode="markers", marker_color=COLOR_OK))
            fig.add_scatter(x=osm, y=np.polyval(np.polyfit(osm, osr, 1), osm), mode="lines", line_color=COLOR_A)
            fig.update_layout(title="Gráfico Q-Q normal de residuos", xaxis_title="Cuantiles teóricos", yaxis_title="Residuos", showlegend=False)
            mostrar_plotly(fig, key="rs_qq")
        if n >= 3:
            w, pw = stats.shapiro(e)
            st.write(f"Prueba de normalidad de Shapiro-Wilk: W = {w:.4f}, p-valor = {pw:.4f} → "
                     f"{'no se rechaza la normalidad ✅' if pw >= alpha else 'los residuos no parecen normales ⚠️'}")
    with t5:
        x0 = st.number_input(f"Valor de {nx} para predecir", value=float(np.round(xb, 2)), key="rs_x0")
        y0 = b0 + b1 * x0
        hm = tc * s * math.sqrt(1 / n + (x0 - xb) ** 2 / Sxx)
        hp = tc * s * math.sqrt(1 + 1 / n + (x0 - xb) ** 2 / Sxx)
        st.latex(rf"\hat{{y}}_0 = {fmt_num(b0)} + ({fmt_num(b1)})({fmt_num(x0)}) = {fmt_num(y0)}")
        st.latex(rf"IC_{{media}}: \hat y_0 \pm t\,s\sqrt{{\tfrac{{1}}{{n}} + \tfrac{{(x_0-\bar x)^2}}{{S_{{xx}}}}}} = ({fmt_num(y0 - hm)},\; {fmt_num(y0 + hm)})")
        st.latex(rf"IP_{{individual}}: \hat y_0 \pm t\,s\sqrt{{1+\tfrac{{1}}{{n}} + \tfrac{{(x_0-\bar x)^2}}{{S_{{xx}}}}}} = ({fmt_num(y0 - hp)},\; {fmt_num(y0 + hp)})")
        if x0 < x.min() or x0 > x.max():
            st.warning("Estás extrapolando (x₀ fuera del rango de los datos): la predicción puede no ser confiable.")

    codigo_r(f"""
x <- c({', '.join(fmt_num(v, 8) for v in x)})
y <- c({', '.join(fmt_num(v, 8) for v in y)})
modelo <- lm(y ~ x)
summary(modelo)                 # coeficientes, R², prueba t y F
anova(modelo)
confint(modelo, level = {1 - alpha})
cor(x, y)
plot(x, y); abline(modelo, col = "red")
plot(fitted(modelo), resid(modelo)); abline(h = 0, lty = 2)
predict(modelo, data.frame(x = {fmt_num(xb)}), interval = "confidence")
predict(modelo, data.frame(x = {fmt_num(xb)}), interval = "prediction")
""")
    reporte(f"El modelo estimado es {ny} = {fmt_num(b0)} {signo} {fmt_num(abs(b1))}·{nx}. Por cada unidad adicional de {nx}, "
            f"{ny} {'aumenta' if b1 >= 0 else 'disminuye'} en promedio {fmt_num(abs(b1))} unidades. El coeficiente de determinación "
            f"R² = {R2 * 100:.2f}% indica la proporción de la variabilidad de {ny} explicada por el modelo, y la correlación es r = {fmt_num(r)}. "
            f"La pendiente {'es' if p_b1 < alpha else 'no es'} estadísticamente significativa (p-valor = {p_b1:.4f}, α = {alpha}).",
            "rs", "Regresión simple")


def _multiple(lector):
    fuente = st.radio("Fuente", ["Tabla editable", "Archivo CSV o Excel"], horizontal=True, key="rm_fuente")
    if fuente == "Tabla editable":
        base = pd.DataFrame({
            "Ventas": [24.0, 30.1, 27.5, 35.2, 40.3, 38.0, 45.6, 48.1, 52.3, 55.0, 60.4, 58.2],
            "Publicidad": [2.0, 3.0, 2.5, 4.0, 5.0, 4.5, 6.0, 6.5, 7.0, 7.5, 8.5, 8.0],
            "Precio": [10.0, 9.8, 10.2, 9.5, 9.0, 9.4, 8.8, 8.5, 8.2, 8.0, 7.5, 7.9],
            "Vendedores": [3, 4, 3, 5, 6, 5, 7, 7, 8, 8, 9, 9]})
        st.caption("Puedes pegar datos desde Excel, agregar filas y renombrar escribiendo un archivo.")
        df = st.data_editor(base, num_rows="dynamic", key="rm_ed")
    else:
        archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key="rm_file")
        if archivo is None:
            return
        df = lector(archivo.getvalue(), archivo.name.lower())
    num = df.select_dtypes("number").columns.tolist()
    if len(num) < 2:
        st.error("Se necesitan columnas numéricas.")
        return
    c1, c2, c3 = st.columns([1, 2, 1])
    yname = c1.selectbox("Variable dependiente Y", num, key="rm_y")
    xnames = c2.multiselect("Variables independientes X", [c for c in num if c != yname],
                            default=[c for c in num if c != yname][:2], key="rm_x")
    alpha = c3.select_slider("α", [0.01, 0.05, 0.10], 0.05, key="rm_alpha")
    if not xnames:
        st.info("Elige al menos una variable X.")
        return
    d = df[[yname] + xnames].dropna().astype(float)
    n, k = len(d), len(xnames)
    if n <= k + 1:
        st.error(f"Se necesitan más de {k + 1} observaciones completas.")
        return
    y = d[yname].to_numpy()
    X = np.column_stack([np.ones(n), d[xnames].to_numpy()])
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < XtX.shape[0]:
        st.error("Las variables X son linealmente dependientes (multicolinealidad perfecta). Quita alguna.")
        return
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    yhat = X @ beta
    e = y - yhat
    SSE = float(e @ e)
    SST = float(np.sum((y - y.mean()) ** 2))
    SSR = SST - SSE
    gl_e = n - k - 1
    MSE = SSE / gl_e
    s = math.sqrt(MSE)
    R2 = SSR / SST
    R2a = 1 - (1 - R2) * (n - 1) / gl_e
    se = np.sqrt(np.diag(XtX_inv) * MSE)
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), gl_e)
    tc = stats.t.ppf(1 - alpha / 2, gl_e)
    F = (SSR / k) / MSE
    pF = stats.f.sf(F, k, gl_e)

    nombres = ["Intercepto"] + xnames
    ecuacion = rf"\hat{{y}} = {fmt_num(beta[0])}" + "".join(
        rf" {'+' if b >= 0 else '-'} {fmt_num(abs(b))}\,\text{{{nm}}}" for b, nm in zip(beta[1:], xnames))
    st.latex(ecuacion)
    c = st.columns(5)
    c[0].metric("R²", f"{R2 * 100:.2f}%")
    c[1].metric("R² ajustado", f"{R2a * 100:.2f}%")
    c[2].metric("F global", fmt_num(F))
    c[3].metric("p-valor F", f"{pF:.2e}")
    c[4].metric("s (error estándar)", fmt_num(s))

    t1, t2, t3, t4, t5 = st.tabs(["🧪 Coeficientes", "📊 ANOVA", "📐 Forma matricial", "🔍 Diagnóstico", "🔮 Predicción"])
    with t1:
        coef = pd.DataFrame({"Variable": nombres, "β estimado": beta, "Error estándar": se, "t": tvals, "p-valor": pvals,
                             "IC inf": beta - tc * se, "IC sup": beta + tc * se,
                             "¿Significativo?": ["✅" if p < alpha else "❌" for p in pvals]})
        st.dataframe(coef.style.format(precision=5), hide_index=True)
        for b, nm, p in zip(beta[1:], xnames, pvals[1:]):
            st.markdown(f"- **{nm}**: manteniendo constantes las demás variables, por cada unidad adicional {yname} "
                        f"{'aumenta' if b >= 0 else 'disminuye'} en promedio {fmt_num(abs(b))} "
                        f"({'significativo' if p < alpha else 'no significativo'}, p = {p:.4f}).")
    with t2:
        st.latex(r"H_0:\beta_1=\cdots=\beta_k=0 \quad vs\quad H_1:\text{al menos un }\beta_j\neq 0")
        st.dataframe(pd.DataFrame({"Fuente": ["Regresión", "Error", "Total"], "SC": [SSR, SSE, SST], "gl": [k, gl_e, n - 1],
                                   "CM": [SSR / k, MSE, None], "F": [F, None, None], "p-valor": [pF, None, None]})
                     .style.format(precision=5, na_rep=""), hide_index=True)
        st.latex(rf"F = \frac{{SSR/k}}{{SSE/(n-k-1)}} = \frac{{{fmt_num(SSR)}/{k}}}{{{fmt_num(SSE)}/{gl_e}}} = {fmt_num(F)}")
        st.latex(rf"R^2_{{aj}} = 1-(1-R^2)\frac{{n-1}}{{n-k-1}} = {fmt_num(R2a)}")
        (st.success if pF < alpha else st.warning)(
            f"{'Se rechaza' if pF < alpha else 'No se rechaza'} H₀: el modelo {'es' if pF < alpha else 'no es'} globalmente significativo.")
    with t3:
        st.latex(r"\hat{\boldsymbol\beta} = (X^TX)^{-1}X^T\mathbf{y}")
        c1, c2 = st.columns(2)
        c1.markdown("**XᵀX**")
        c1.dataframe(pd.DataFrame(XtX, index=nombres, columns=nombres).style.format(precision=3))
        c2.markdown("**Xᵀy**")
        c2.dataframe(pd.DataFrame(X.T @ y, index=nombres, columns=["Xᵀy"]).style.format(precision=3))
        st.markdown("**(XᵀX)⁻¹**")
        st.dataframe(pd.DataFrame(XtX_inv, index=nombres, columns=nombres).style.format(precision=6))
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(x=y, y=yhat, mode="markers", marker_color=COLOR_PRIMARIO))
            fig.add_scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode="lines", line=dict(dash="dash", color=COLOR_A))
            fig.update_layout(title="Observado vs predicho", xaxis_title="y observado", yaxis_title="ŷ", showlegend=False)
            mostrar_plotly(fig, key="rm_obs")
        with c2:
            fig = go.Figure(go.Scatter(x=yhat, y=e, mode="markers", marker_color=COLOR_OK))
            fig.add_hline(y=0, line_dash="dash", line_color=COLOR_A)
            fig.update_layout(title="Residuos vs ajustados", xaxis_title="ŷ", yaxis_title="e")
            mostrar_plotly(fig, key="rm_res")
        if k >= 2:
            vif = []
            for j in range(k):
                otras = np.column_stack([np.ones(n), np.delete(d[xnames].to_numpy(), j, axis=1)])
                yj = d[xnames[j]].to_numpy()
                bj = np.linalg.lstsq(otras, yj, rcond=None)[0]
                r2j = 1 - np.sum((yj - otras @ bj) ** 2) / np.sum((yj - yj.mean()) ** 2)
                vif.append(1 / (1 - r2j) if r2j < 1 else float("inf"))
            st.markdown("**Multicolinealidad (VIF)** — valores > 10 indican problema")
            st.dataframe(pd.DataFrame({"Variable": xnames, "VIF": vif}).style.format(precision=3), hide_index=True)
        corr = d.corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmin=-1, zmax=1, texttemplate="%{z:.2f}"))
        fig.update_layout(title="Matriz de correlación", yaxis_autorange="reversed")
        mostrar_plotly(fig, key="rm_corr")
        if k == 2:
            g1 = np.linspace(d[xnames[0]].min(), d[xnames[0]].max(), 20)
            g2 = np.linspace(d[xnames[1]].min(), d[xnames[1]].max(), 20)
            G1, G2 = np.meshgrid(g1, g2)
            Z = beta[0] + beta[1] * G1 + beta[2] * G2
            fig = go.Figure([go.Surface(x=G1, y=G2, z=Z, opacity=0.6, colorscale="Blues", showscale=False),
                             go.Scatter3d(x=d[xnames[0]], y=d[xnames[1]], z=y, mode="markers", marker=dict(size=5, color=COLOR_A))])
            fig.update_layout(title="Plano de regresión (gíralo con el mouse)", height=560,
                              scene=dict(xaxis_title=xnames[0], yaxis_title=xnames[1], zaxis_title=yname))
            mostrar_plotly(fig, key="rm_3d")
    with t5:
        cols = st.columns(k)
        x0 = [1.0] + [cols[j].number_input(xnames[j], value=float(d[xnames[j]].mean()), key=f"rm_p_{xnames[j]}") for j in range(k)]
        x0 = np.array(x0)
        y0 = float(x0 @ beta)
        h = float(x0 @ XtX_inv @ x0)
        st.latex(rf"\hat y_0 = \mathbf{{x}}_0^T\hat{{\boldsymbol\beta}} = {fmt_num(y0)}")
        st.write(f"IC {100 * (1 - alpha):.0f}% para la media: ({fmt_num(y0 - tc * s * math.sqrt(h))}, {fmt_num(y0 + tc * s * math.sqrt(h))})")
        st.write(f"IP {100 * (1 - alpha):.0f}% individual: ({fmt_num(y0 - tc * s * math.sqrt(1 + h))}, {fmt_num(y0 + tc * s * math.sqrt(1 + h))})")

    formula = " + ".join(f"`{x}`" if " " in x else x for x in xnames)
    codigo_r(f"""
datos <- read.csv("tus_datos.csv")      # o usa readxl::read_excel()
modelo <- lm({yname} ~ {formula}, data = datos)
summary(modelo)                          # coeficientes, R², R² ajustado y F
anova(modelo)
confint(modelo, level = {1 - alpha})
car::vif(modelo)                         # multicolinealidad (paquete car)
par(mfrow = c(2, 2)); plot(modelo)       # diagnóstico de residuos
predict(modelo, newdata = data.frame({', '.join(f'{x} = {fmt_num(d[x].mean())}' for x in xnames)}), interval = "prediction")
""")
    reporte(f"El modelo de regresión múltiple para {yname} con {k} predictores ({', '.join(xnames)}) explica el {R2 * 100:.2f}% de la "
            f"variabilidad (R² ajustado = {R2a * 100:.2f}%). El modelo {'es' if pF < alpha else 'no es'} globalmente significativo "
            f"(F = {fmt_num(F)}, p = {pF:.4g}). Variables significativas al {alpha}: "
            f"{', '.join(nm for nm, p in zip(xnames, pvals[1:]) if p < alpha) or 'ninguna'}.", "rm", "Regresión múltiple")
