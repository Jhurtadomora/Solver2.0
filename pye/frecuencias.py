"""CA1 · Tablas de frecuencia (no agrupadas, agrupadas en intervalos y cualitativas)."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .comunes import (COLOR_ACENTO, COLOR_OK, COLOR_PRIMARIO, a_float, codigo_r, descargas_tabla,
                      fmt_num, leer_columna, mostrar_plotly, r_vector, reporte, separar_tokens)

EJEMPLO = ("23, 25, 31, 28, 35, 41, 29, 33, 37, 26, 30, 45, 38, 27, 32, 34, 29, 36, 40, 31, "
           "28, 33, 39, 42, 30, 35, 27, 31, 34, 38")


def _agregar_totales(df, columnas_suma, etiqueta_col):
    total = {c: "" for c in df.columns}
    total[etiqueta_col] = "Total"
    for c in columnas_suma:
        total[c] = df[c].sum()
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


def _obtener_datos(lector):
    fuente = st.radio("Fuente de datos", ["Datos manuales", "Archivo CSV o Excel"], horizontal=True, key="tf_fuente")
    if fuente == "Datos manuales":
        c1, c2 = st.columns([3, 1])
        txt = c1.text_area("Datos (separados por coma, espacio, salto de línea o punto y coma)", EJEMPLO,
                           key="tf_txt", height=110)
        nombre = c2.text_input("Nombre de la variable", "Edad", key="tf_nombre")
        tokens = separar_tokens(txt)
        try:
            datos = [a_float(t) for t in tokens]
            return datos, nombre, True
        except Exception:
            return tokens, nombre, False
    serie, nombre, _ = leer_columna(lector, "tf")
    if serie is None:
        return [], None, True
    serie = serie.dropna()
    numerica = pd.to_numeric(serie.astype(str).str.replace(",", ".", regex=False), errors="coerce")
    if len(serie) and numerica.notna().mean() > 0.9:
        return numerica.dropna().tolist(), nombre, True
    return serie.astype(str).tolist(), nombre, False


def render_tabla_frecuencias(lector):
    datos, nombre, es_numerica = _obtener_datos(lector)
    if len(datos) < 2:
        st.info("Ingresa al menos 2 datos para construir la tabla de frecuencias.")
        return

    opciones = (["Datos no agrupados (variable discreta)", "Datos agrupados en intervalos (variable continua)",
                 "Cualitativa (categorías)"] if es_numerica else ["Cualitativa (categorías)"])
    sugerida = 0 if es_numerica and len(set(datos)) <= 15 else (1 if es_numerica else 0)
    tipo = st.radio("Tipo de tabla", opciones, index=sugerida, horizontal=True, key="tf_tipo",
                    help="Sugerencia automática: si hay muchos valores distintos conviene agrupar en intervalos.")
    st.caption(f"n = {len(datos)} datos · {len(set(datos))} valores distintos")

    if tipo.startswith("Datos no agrupados"):
        _tabla_no_agrupada(datos, nombre)
    elif tipo.startswith("Datos agrupados"):
        _tabla_agrupada(datos, nombre)
    else:
        _tabla_cualitativa([str(d) if not isinstance(d, str) else d for d in datos], nombre)


# ------------------------------------------------------------------
# No agrupada
# ------------------------------------------------------------------

def _tabla_no_agrupada(datos, nombre):
    s = pd.Series(datos)
    n = len(s)
    vc = s.value_counts().sort_index()
    df = pd.DataFrame({"x_i": vc.index.astype(float), "f_i": vc.values.astype(int)})
    df["F_i"] = df["f_i"].cumsum()
    df["h_i"] = df["f_i"] / n
    df["H_i"] = df["h_i"].cumsum()
    df["% h_i"] = df["h_i"] * 100
    df["% H_i"] = df["H_i"] * 100
    df["x_i·f_i"] = df["x_i"] * df["f_i"]
    df["x_i²·f_i"] = df["x_i"] ** 2 * df["f_i"]

    st.markdown("### 📋 Tabla de frecuencias")
    vista = _agregar_totales(df, ["f_i", "h_i", "% h_i", "x_i·f_i", "x_i²·f_i"], "x_i")
    st.dataframe(vista.style.format(precision=4, na_rep=""), hide_index=True)
    st.caption("f_i: frecuencia absoluta · F_i: absoluta acumulada · h_i: relativa · H_i: relativa acumulada")
    descargas_tabla(df, f"tabla_frecuencias_{nombre}", "tf_na")

    media = df["x_i·f_i"].sum() / n
    var_m = (df["x_i²·f_i"].sum() - n * media ** 2) / (n - 1)
    moda_vals = df.loc[df["f_i"] == df["f_i"].max(), "x_i"].tolist()
    mediana = float(np.median(datos))

    c = st.columns(4)
    c[0].metric("Media x̄", fmt_num(media))
    c[1].metric("Mediana", fmt_num(mediana))
    c[2].metric("Moda", ", ".join(fmt_num(m) for m in moda_vals[:4]))
    c[3].metric("Desv. estándar s", fmt_num(math.sqrt(max(var_m, 0))))

    with st.expander("📐 Demostración: indicadores calculados desde la tabla"):
        st.latex(rf"\bar{{x}} = \frac{{\sum x_i f_i}}{{n}} = \frac{{{fmt_num(df['x_i·f_i'].sum())}}}{{{n}}} = {fmt_num(media)}")
        st.latex(rf"s^2 = \frac{{\sum x_i^2 f_i - n\bar{{x}}^2}}{{n-1}} = \frac{{{fmt_num(df['x_i²·f_i'].sum())} - {n}({fmt_num(media)})^2}}{{{n - 1}}} = {fmt_num(var_m)}")
        st.latex(rf"s = \sqrt{{{fmt_num(var_m)}}} = {fmt_num(math.sqrt(max(var_m, 0)))}")
        pos = (n + 1) / 2
        st.markdown(f"**Mediana:** la posición central es (n+1)/2 = {fmt_num(pos)}; se busca el primer valor con "
                    f"F_i ≥ {fmt_num(n / 2)} → **{fmt_num(mediana)}**.")
        st.markdown(f"**Moda:** valor(es) con mayor f_i = {int(df['f_i'].max())} → **{', '.join(fmt_num(m) for m in moda_vals)}**.")

    st.markdown("### 📊 Gráficos")
    medida = st.radio("Frecuencia a graficar", ["Absoluta (f_i)", "Relativa (h_i)", "Porcentaje"], horizontal=True, key="tf_na_med")
    col_y = {"Absoluta (f_i)": "f_i", "Relativa (h_i)": "h_i", "Porcentaje": "% h_i"}[medida]
    col_acum = {"f_i": "F_i", "h_i": "H_i", "% h_i": "% H_i"}[col_y]
    t1, t2, t3 = st.tabs(["Barras", "Acumulada (escalonada)", "Torta"])
    with t1:
        fig = go.Figure(go.Bar(x=df["x_i"], y=df[col_y], marker_color=COLOR_PRIMARIO,
                               text=[fmt_num(v) for v in df[col_y]], textposition="outside",
                               hovertemplate="x = %{x}<br>" + col_y + " = %{y}<extra></extra>"))
        fig.update_layout(title=f"Diagrama de barras de {nombre}", xaxis_title=nombre, yaxis_title=col_y, bargap=0.5)
        mostrar_plotly(fig, key="tf_na_bar")
    with t2:
        xs = [df["x_i"].min() - 1] + df["x_i"].tolist() + [df["x_i"].max() + 1]
        ys = [0] + df[col_acum].tolist() + [df[col_acum].iloc[-1]]
        fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", line_shape="hv", line_color=COLOR_OK,
                                   hovertemplate="x = %{x}<br>acumulada = %{y}<extra></extra>"))
        fig.update_layout(title=f"Frecuencia acumulada de {nombre}", xaxis_title=nombre, yaxis_title=col_acum)
        mostrar_plotly(fig, key="tf_na_acum")
    with t3:
        fig = go.Figure(go.Pie(labels=[fmt_num(v) for v in df["x_i"]], values=df["f_i"], hole=0.35))
        fig.update_layout(title=f"Composición de {nombre}")
        mostrar_plotly(fig, key="tf_na_pie")

    codigo_r(f"""
x <- {r_vector(datos)}
tabla <- table(x)
fi <- as.vector(tabla)
hi <- fi / sum(fi)
tf <- data.frame(xi = as.numeric(names(tabla)), fi = fi, Fi = cumsum(fi),
                 hi = round(hi, 4), Hi = round(cumsum(hi), 4))
tf
mean(x); median(x); sd(x)
barplot(tabla, main = "Diagrama de barras", xlab = "{nombre}")
""")
    reporte(f"La variable {nombre} tiene {n} observaciones con {len(df)} valores distintos. "
            f"El valor más frecuente es {', '.join(fmt_num(m) for m in moda_vals)} (f = {int(df['f_i'].max())}), "
            f"la media es {fmt_num(media)} y la mediana {fmt_num(mediana)}. "
            f"El {fmt_num(df['% H_i'].iloc[len(df) // 2], 2)}% de los datos es menor o igual a {fmt_num(df['x_i'].iloc[len(df) // 2])}.",
            "tf_na", "Tabla de frecuencias")


# ------------------------------------------------------------------
# Agrupada en intervalos
# ------------------------------------------------------------------

def _construir_clases(x, k, A, L0, cerrado_izq):
    edges = np.round(L0 + A * np.arange(k + 1), 10)
    while edges[-1] < x.max() - 1e-12:
        edges = np.append(edges, np.round(edges[-1] + A, 10))
    if cerrado_izq:
        idx = np.searchsorted(edges, x, side="right") - 1
        idx = np.where(x >= edges[-1], len(edges) - 2, idx)
    else:
        idx = np.searchsorted(edges, x, side="left") - 1
        idx = np.where(x <= edges[0], 0, idx)
    conteos = np.bincount(idx, minlength=len(edges) - 1)[: len(edges) - 1]
    return edges, conteos


def _tabla_agrupada(datos, nombre):
    x = np.array(datos, dtype=float)
    n = len(x)
    xmin, xmax = x.min(), x.max()
    R = xmax - xmin

    st.markdown("### ⚙️ Construcción de las clases")
    c1, c2, c3, c4 = st.columns(4)
    regla = c1.selectbox("Número de clases k", ["Regla de Sturges", "Raíz cuadrada de n", "Manual"], key="tf_regla")
    k_sturges = 1 + 3.322 * math.log10(n)
    k_raiz = math.sqrt(n)
    if regla == "Regla de Sturges":
        k = math.ceil(k_sturges)
    elif regla == "Raíz cuadrada de n":
        k = math.ceil(k_raiz)
    else:
        k = int(c1.number_input("k", 2, 50, max(2, math.ceil(k_sturges)), key="tf_k_manual"))
    dec = int(c2.number_input("Decimales de la amplitud", 0, 4, 0 if np.all(x == np.round(x)) else 1, key="tf_dec",
                              help="La amplitud se redondea hacia ARRIBA con estos decimales."))
    A_cruda = R / k if R > 0 else 1
    A_auto = math.ceil(A_cruda * 10 ** dec - 1e-9) / 10 ** dec or 1 / 10 ** dec
    usar_A_manual = c3.checkbox("Amplitud manual", key="tf_A_manual_chk")
    A = c3.number_input("Amplitud A", min_value=1e-6, value=float(A_auto), key="tf_A_manual") if usar_A_manual else A_auto
    L0 = c4.number_input("Límite inferior 1ª clase", value=float(xmin), key="tf_L0")
    cerrado_izq = st.radio("Tipo de intervalo", ["[Lᵢ, Lₛ)  cerrado a la izquierda", "(Lᵢ, Lₛ]  cerrado a la derecha"],
                           horizontal=True, key="tf_cierre").startswith("[")
    if L0 > xmin:
        st.warning(f"El límite inferior ({fmt_num(L0)}) es mayor que el dato mínimo ({fmt_num(xmin)}); se usa el mínimo.")
        L0 = xmin

    edges, f = _construir_clases(x, k, A, L0, cerrado_izq)
    k_real = len(f)
    if k_real != k:
        st.info(f"Se agregaron clases para cubrir el dato máximo: k final = {k_real}.")

    with st.expander("📐 Demostración: cálculo de k, rango y amplitud", expanded=False):
        st.latex(rf"R = x_{{max}} - x_{{min}} = {fmt_num(xmax)} - {fmt_num(xmin)} = {fmt_num(R)}")
        st.latex(rf"k_{{Sturges}} = 1 + 3.322\log_{{10}}(n) = 1 + 3.322\log_{{10}}({n}) = {k_sturges:.4f} \approx {math.ceil(k_sturges)}")
        st.latex(rf"k_{{\sqrt{{n}}}} = \sqrt{{{n}}} = {k_raiz:.4f} \approx {math.ceil(k_raiz)}")
        st.latex(rf"A = \frac{{R}}{{k}} = \frac{{{fmt_num(R)}}}{{{k}}} = {A_cruda:.4f} \;\Rightarrow\; A = {fmt_num(A)}")

    Li, Ls = edges[:-1], edges[1:]
    marcas = (Li + Ls) / 2
    df = pd.DataFrame({
        "Clase": np.arange(1, k_real + 1),
        "Intervalo": [(f"[{fmt_num(a)}, {fmt_num(b)})" if cerrado_izq else f"({fmt_num(a)}, {fmt_num(b)}]") for a, b in zip(Li, Ls)],
        "Lᵢ": Li, "Lₛ": Ls, "Marca x_i": marcas, "f_i": f.astype(int),
    })
    if cerrado_izq and k_real:
        df.loc[k_real - 1, "Intervalo"] = f"[{fmt_num(Li[-1])}, {fmt_num(Ls[-1])}]"
    df["F_i"] = df["f_i"].cumsum()
    df["h_i"] = df["f_i"] / n
    df["H_i"] = df["h_i"].cumsum()
    df["% h_i"] = df["h_i"] * 100
    df["x_i·f_i"] = df["Marca x_i"] * df["f_i"]
    media = df["x_i·f_i"].sum() / n
    df["(x_i−x̄)²·f_i"] = (df["Marca x_i"] - media) ** 2 * df["f_i"]

    st.markdown("### 📋 Tabla de frecuencias agrupada")
    vista = _agregar_totales(df, ["f_i", "h_i", "% h_i", "x_i·f_i", "(x_i−x̄)²·f_i"], "Intervalo")
    st.dataframe(vista.drop(columns=["Lᵢ", "Lₛ"]).style.format(precision=4, na_rep=""), hide_index=True)
    descargas_tabla(df, f"tabla_agrupada_{nombre}", "tf_ag")

    # Indicadores agrupados
    tipo_var = st.radio("Varianza", ["Muestral (n−1)", "Poblacional (N)"], horizontal=True, key="tf_ag_var")
    den = n - 1 if tipo_var.startswith("Muestral") else n
    var = df["(x_i−x̄)²·f_i"].sum() / den
    F_prev = np.concatenate([[0], df["F_i"].values[:-1]])

    def cuantil(prop):
        objetivo = prop * n
        i = int(np.argmax(df["F_i"].values >= objetivo - 1e-12))
        valor = Li[i] + ((objetivo - F_prev[i]) / f[i]) * A if f[i] > 0 else Li[i]
        return valor, i, objetivo

    me, i_me, obj_me = cuantil(0.5)
    i_mo = int(np.argmax(f))
    d1 = f[i_mo] - (f[i_mo - 1] if i_mo > 0 else 0)
    d2 = f[i_mo] - (f[i_mo + 1] if i_mo < k_real - 1 else 0)
    mo = Li[i_mo] + (d1 / (d1 + d2)) * A if (d1 + d2) != 0 else marcas[i_mo]

    st.markdown("### 🧮 Indicadores para datos agrupados")
    c = st.columns(5)
    c[0].metric("Media x̄", fmt_num(media))
    c[1].metric("Mediana Me", fmt_num(me))
    c[2].metric("Moda Mo", fmt_num(mo))
    c[3].metric("Varianza", fmt_num(var))
    c[4].metric("Desv. estándar", fmt_num(math.sqrt(var)))
    st.caption(f"Comparación con los datos sin agrupar: media real = {fmt_num(x.mean())}, mediana real = {fmt_num(np.median(x))}. "
               "La diferencia se debe a que al agrupar se usa la marca de clase como representante.")

    pk = st.slider("Percentil Pₖ a calcular", 1, 99, 75, key="tf_pk")
    p_val, i_p, obj_p = cuantil(pk / 100)
    st.metric(f"P{pk}", fmt_num(p_val))

    with st.expander("📐 Demostración paso a paso (fórmulas de datos agrupados)", expanded=True):
        st.markdown("**Media**")
        st.latex(rf"\bar{{x}} = \frac{{\sum x_i f_i}}{{n}} = \frac{{{fmt_num(df['x_i·f_i'].sum())}}}{{{n}}} = {fmt_num(media)}")
        st.markdown(f"**Mediana** — clase mediana: la primera con F_i ≥ n/2 = {fmt_num(obj_me)} → clase {i_me + 1} {df['Intervalo'][i_me]}")
        st.latex(rf"Me = L_i + \frac{{\frac{{n}}{{2}} - F_{{i-1}}}}{{f_i}}\cdot A = {fmt_num(Li[i_me])} + \frac{{{fmt_num(obj_me)} - {int(F_prev[i_me])}}}{{{int(f[i_me])}}}\cdot {fmt_num(A)} = {fmt_num(me)}")
        st.markdown(f"**Moda** — clase modal: la de mayor f_i → clase {i_mo + 1} {df['Intervalo'][i_mo]}")
        st.latex(rf"Mo = L_i + \frac{{d_1}}{{d_1 + d_2}}\cdot A,\quad d_1 = f_i - f_{{i-1}} = {int(d1)},\; d_2 = f_i - f_{{i+1}} = {int(d2)}")
        st.latex(rf"Mo = {fmt_num(Li[i_mo])} + \frac{{{int(d1)}}}{{{int(d1 + d2)}}}\cdot {fmt_num(A)} = {fmt_num(mo)}")
        st.markdown("**Varianza**")
        st.latex(rf"s^2 = \frac{{\sum (x_i-\bar{{x}})^2 f_i}}{{{'n-1' if den == n - 1 else 'N'}}} = \frac{{{fmt_num(df['(x_i−x̄)²·f_i'].sum())}}}{{{den}}} = {fmt_num(var)}")
        st.markdown(f"**Percentil {pk}** — posición k·n/100 = {fmt_num(obj_p)} → clase {i_p + 1} {df['Intervalo'][i_p]}")
        st.latex(rf"P_{{{pk}}} = L_i + \frac{{\frac{{{pk}\,n}}{{100}} - F_{{i-1}}}}{{f_i}}\cdot A = {fmt_num(Li[i_p])} + \frac{{{fmt_num(obj_p)} - {int(F_prev[i_p])}}}{{{int(f[i_p])}}}\cdot {fmt_num(A)} = {fmt_num(p_val)}")

    st.markdown("### 📊 Gráficos")
    medida = st.radio("Escala", ["Absoluta", "Relativa", "Porcentaje"], horizontal=True, key="tf_ag_med")
    factor = {"Absoluta": 1, "Relativa": 1 / n, "Porcentaje": 100 / n}[medida]
    t1, t2 = st.tabs(["Histograma + polígono", "Ojiva"])
    with t1:
        mostrar_pol = st.checkbox("Mostrar polígono de frecuencias", True, key="tf_pol")
        fig = go.Figure(go.Bar(x=marcas, y=f * factor, width=[A] * k_real, marker_color=COLOR_PRIMARIO,
                               marker_line_color="white", marker_line_width=1, name="Histograma",
                               customdata=df["Intervalo"],
                               hovertemplate="%{customdata}<br>frecuencia = %{y}<extra></extra>"))
        if mostrar_pol:
            px_ = np.concatenate([[marcas[0] - A], marcas, [marcas[-1] + A]])
            py_ = np.concatenate([[0], f * factor, [0]])
            fig.add_trace(go.Scatter(x=px_, y=py_, mode="lines+markers", name="Polígono", line_color=COLOR_ACENTO))
        for valor, nom, color in [(media, "Media", COLOR_OK), (me, "Mediana", "#a855f7")]:
            fig.add_vline(x=valor, line_dash="dash", line_color=color, annotation_text=f"{nom} = {fmt_num(valor, 2)}")
        fig.update_layout(title=f"Histograma de {nombre}", xaxis_title=nombre, yaxis_title=f"Frecuencia {medida.lower()}",
                          xaxis=dict(tickmode="array", tickvals=edges, ticktext=[fmt_num(e, 2) for e in edges]))
        mostrar_plotly(fig, key="tf_ag_hist")
    with t2:
        fig = go.Figure(go.Scatter(x=edges, y=np.concatenate([[0], df["F_i"].values * factor]), mode="lines+markers",
                                   line_color=COLOR_OK, hovertemplate="x = %{x}<br>acumulada = %{y}<extra></extra>"))
        fig.add_hline(y=n * factor / 2, line_dash="dot", annotation_text="50% → mediana")
        fig.update_layout(title=f"Ojiva (frecuencia acumulada) de {nombre}", xaxis_title=nombre, yaxis_title="Acumulada")
        mostrar_plotly(fig, key="tf_ag_ojiva")

    codigo_r(f"""
x <- {r_vector(datos)}
k <- nclass.Sturges(x)          # regla de Sturges
cortes <- seq({fmt_num(edges[0], 6)}, by = {fmt_num(A, 6)}, length.out = {k_real + 1})
clases <- cut(x, breaks = cortes, right = {"FALSE" if cerrado_izq else "TRUE"}, include.lowest = TRUE)
fi <- as.vector(table(clases))
tf <- data.frame(Clase = levels(clases), xi = (head(cortes, -1) + tail(cortes, -1)) / 2,
                 fi = fi, Fi = cumsum(fi), hi = round(fi / sum(fi), 4), Hi = round(cumsum(fi) / sum(fi), 4))
tf
media_agrupada <- sum(tf$xi * tf$fi) / sum(tf$fi)
hist(x, breaks = cortes, right = {"FALSE" if cerrado_izq else "TRUE"}, main = "Histograma", xlab = "{nombre}")
""")
    reporte(f"Los {n} datos de {nombre} se agruparon en {k_real} clases de amplitud {fmt_num(A)}. "
            f"La clase con mayor frecuencia es {df['Intervalo'][i_mo]} (f = {int(f[i_mo])}). "
            f"Con datos agrupados, la media es {fmt_num(media)}, la mediana {fmt_num(me)} y la moda {fmt_num(mo)}; "
            f"la desviación estándar es {fmt_num(math.sqrt(var))}. El {pk}% de los datos es menor o igual a {fmt_num(p_val)}.",
            "tf_ag", "Tabla de frecuencias agrupada")


# ------------------------------------------------------------------
# Cualitativa
# ------------------------------------------------------------------

def _tabla_cualitativa(datos, nombre):
    s = pd.Series(datos)
    n = len(s)
    c1, c2 = st.columns(2)
    orden = c1.selectbox("Orden de las categorías", ["Mayor a menor frecuencia", "Alfabético", "Orden de aparición", "Orden personalizado"], key="tf_cual_orden")
    ordinal = c2.checkbox("La variable es ordinal (tiene sentido acumular)", key="tf_ordinal")
    vc = s.value_counts()
    if orden == "Alfabético":
        vc = vc.sort_index()
    elif orden == "Orden de aparición":
        vc = vc.reindex(pd.unique(s))
    elif orden == "Orden personalizado":
        cats = st.multiselect("Ordena las categorías (en el orden en que las selecciones)", list(vc.index), default=list(vc.index), key="tf_cual_custom")
        resto = [c for c in vc.index if c not in cats]
        vc = vc.reindex(cats + resto)
    df = pd.DataFrame({"Categoría": vc.index, "f_i": vc.values.astype(int)})
    df["h_i"] = df["f_i"] / n
    df["% h_i"] = df["h_i"] * 100
    if ordinal or orden == "Mayor a menor frecuencia":
        df["F_i"] = df["f_i"].cumsum()
        df["H_i"] = df["h_i"].cumsum()
        df["% H_i"] = df["H_i"] * 100
    st.markdown("### 📋 Tabla de frecuencias")
    st.dataframe(_agregar_totales(df, ["f_i", "h_i", "% h_i"], "Categoría").style.format(precision=4, na_rep=""), hide_index=True)
    descargas_tabla(df, f"tabla_cualitativa_{nombre}", "tf_cu")
    moda = df.loc[df["f_i"] == df["f_i"].max(), "Categoría"].tolist()
    c = st.columns(3)
    c[0].metric("Moda", ", ".join(moda[:3]))
    c[1].metric("Categorías", len(df))
    c[2].metric("n", n)

    t1, t2, t3 = st.tabs(["Barras", "Pareto", "Torta"])
    with t1:
        fig = go.Figure(go.Bar(x=df["Categoría"], y=df["f_i"], marker_color=COLOR_PRIMARIO,
                               text=[f"{p:.1f}%" for p in df["% h_i"]], textposition="outside"))
        fig.update_layout(title=f"Frecuencia de {nombre}", yaxis_title="f_i")
        mostrar_plotly(fig, key="tf_cu_bar")
    with t2:
        dp = df.sort_values("f_i", ascending=False)
        acum = dp["f_i"].cumsum() / n * 100
        fig = go.Figure()
        fig.add_bar(x=dp["Categoría"], y=dp["f_i"], name="f_i", marker_color=COLOR_PRIMARIO)
        fig.add_scatter(x=dp["Categoría"], y=acum, name="% acumulado", yaxis="y2", mode="lines+markers", line_color=COLOR_ACENTO)
        fig.add_hline(y=80, line_dash="dot", yref="y2", annotation_text="80%")
        fig.update_layout(title="Diagrama de Pareto", yaxis2=dict(overlaying="y", side="right", range=[0, 105], title="%"))
        mostrar_plotly(fig, key="tf_cu_pareto")
    with t3:
        fig = go.Figure(go.Pie(labels=df["Categoría"], values=df["f_i"], hole=0.35))
        mostrar_plotly(fig, key="tf_cu_pie")

    codigo_r(f"""
x <- {r_vector(datos)}
tabla <- sort(table(x), decreasing = TRUE)
data.frame(categoria = names(tabla), fi = as.vector(tabla),
           hi = round(as.vector(prop.table(tabla)), 4))
barplot(tabla, main = "Frecuencia de {nombre}")
pie(tabla)
""")
    reporte(f"La variable cualitativa {nombre} tiene {n} observaciones distribuidas en {len(df)} categorías. "
            f"La categoría más frecuente (moda) es {', '.join(moda)}, con {int(df['f_i'].max())} casos "
            f"({df['% h_i'].max():.2f}% del total).", "tf_cu", "Tabla de frecuencias cualitativa")
