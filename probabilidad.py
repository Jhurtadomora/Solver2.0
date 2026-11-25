"""CA2 · Probabilidad: operaciones con eventos, Venn, tablas de contingencia y enfoque frecuentista."""
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import to_rgb

from .comunes import (COLOR_A, COLOR_ACENTO, COLOR_B, COLOR_OK, COLOR_PRIMARIO, a_fraccion, codigo_r,
                      fmt_p, latex_num, mostrar_plotly, reporte, tx)


def render_probabilidad():
    tema = st.selectbox("Tema", [
        "Operaciones con dos eventos (A, B) + Venn",
        "Tres eventos (inclusión–exclusión) + Venn",
        "Tabla de contingencia (probabilidad conjunta, marginal y condicional)",
        "Enfoque clásico vs frecuentista (simulación)",
    ], key="prob_tema")
    modo = st.radio("Mostrar probabilidades como", ["Decimal", "Fracción", "Porcentaje"], horizontal=True, key="prob_fmt")
    if tema.startswith("Operaciones"):
        _dos_eventos(modo)
    elif tema.startswith("Tres"):
        _tres_eventos(modo)
    elif tema.startswith("Tabla"):
        _contingencia(modo)
    else:
        _simulacion_frecuentista(modo)


# ------------------------------------------------------------------
# Diagrama de Venn exacto (sombreado por regiones)
# ------------------------------------------------------------------

def dibujar_venn(nombres, valores_region, resaltar, modo):
    """nombres: lista de 2 o 3 nombres. valores_region: dict tupla(bool...)->prob. resaltar: set de tuplas."""
    k = len(nombres)
    centros = [(-0.55, 0), (0.55, 0)] if k == 2 else [(-0.55, 0.35), (0.55, 0.35), (0, -0.55)]
    r = 1.0
    xs = np.linspace(-2.2, 2.2, 500)
    ys = np.linspace(-1.9, 1.7, 420)
    X, Y = np.meshgrid(xs, ys)
    dentro = [((X - cx) ** 2 + (Y - cy) ** 2) <= r ** 2 for cx, cy in centros]
    img = np.ones(X.shape + (3,))
    base = np.array(to_rgb("#e2e8f0"))
    color_res = np.array(to_rgb(COLOR_A))
    for region in valores_region:
        mask = np.ones(X.shape, bool)
        for i, en in enumerate(region):
            mask &= dentro[i] if en else ~dentro[i]
        img[mask] = color_res * 0.75 + 0.25 if region in resaltar else (base if any(region) else np.array([0.97, 0.98, 0.99]))
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(img, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower")
    ax.add_patch(plt.Rectangle((xs[0] + 0.02, ys[0] + 0.02), xs[-1] - xs[0] - 0.04, ys[-1] - ys[0] - 0.04, fill=False, lw=1.5))
    for (cx, cy), nom in zip(centros, nombres):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, lw=2))
    etiquetas = [(-1.2, 0.95), (1.2, 0.95)] if k == 2 else [(-1.35, 1.25), (1.35, 1.25), (0, -1.75)]
    for (lx, ly), nom in zip(etiquetas, nombres):
        ax.text(lx, ly, nom, fontsize=15, fontweight="bold", ha="center")
    # posiciones de los textos de cada región
    if k == 2:
        pos = {(True, False): (-0.95, 0), (False, True): (0.95, 0), (True, True): (0, 0), (False, False): (1.75, -1.6)}
    else:
        pos = {(True, False, False): (-0.95, 0.65), (False, True, False): (0.95, 0.65), (False, False, True): (0, -1.05),
               (True, True, False): (0, 0.8), (True, False, True): (-0.6, -0.35), (False, True, True): (0.6, -0.35),
               (True, True, True): (0, 0.1), (False, False, False): (1.75, -1.6)}
    for region, p in valores_region.items():
        x, y = pos[region]
        ax.text(x, y, fmt_p(p, modo, 3), ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(ys[0], ys[-1])
    ax.axis("off")
    ax.set_title("S (espacio muestral)", loc="left", fontsize=10)
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------
# Dos eventos
# ------------------------------------------------------------------

def _leer_p(label, default, key):
    txt = st.text_input(label, default, key=key, help="Acepta 0.3, 0,3, 30% o 3/10")
    try:
        p = a_fraccion(txt)
    except Exception:
        st.error(f"{label}: valor no válido")
        return None
    if not 0 <= p <= 1:
        st.error(f"{label} debe estar entre 0 y 1")
        return None
    return p


def _dos_eventos(modo):
    c1, c2, c3 = st.columns(3)
    nA = c1.text_input("Nombre del evento A", "A", key="p2_nA")
    nB = c2.text_input("Nombre del evento B", "B", key="p2_nB")
    relacion = c3.selectbox("¿Qué otro dato conoces?", [
        "P(A ∩ B)", "P(A ∪ B)", "P(A | B)", "P(B | A)", "A y B son independientes", "A y B son mutuamente excluyentes"
    ], key="p2_rel")
    c1, c2, c3 = st.columns(3)
    with c1:
        pa = _leer_p(f"P({nA})", "0.5", "p2_pa")
    with c2:
        pb = _leer_p(f"P({nB})", "0.4", "p2_pb")
    dato = None
    if relacion.startswith("P("):
        with c3:
            dato = _leer_p(relacion.replace("A", nA).replace("B", nB), "0.2", "p2_dato")
    if pa is None or pb is None or (relacion.startswith("P(") and dato is None):
        return

    pasos = []
    if relacion == "P(A ∩ B)":
        pab = dato
    elif relacion == "P(A ∪ B)":
        pab = pa + pb - dato
        pasos.append(rf"P(A\cap B) = P(A) + P(B) - P(A\cup B) = {latex_num(pa, modo)} + {latex_num(pb, modo)} - {latex_num(dato, modo)} = {latex_num(pab, modo)}")
    elif relacion == "P(A | B)":
        pab = dato * pb
        pasos.append(rf"P(A\cap B) = P(A|B)\,P(B) = {latex_num(dato, modo)}\cdot {latex_num(pb, modo)} = {latex_num(pab, modo)}")
    elif relacion == "P(B | A)":
        pab = dato * pa
        pasos.append(rf"P(A\cap B) = P(B|A)\,P(A) = {latex_num(dato, modo)}\cdot {latex_num(pa, modo)} = {latex_num(pab, modo)}")
    elif relacion.startswith("A y B son independientes"):
        pab = pa * pb
        pasos.append(rf"\text{{Independientes}} \Rightarrow P(A\cap B) = P(A)P(B) = {latex_num(pa, modo)}\cdot{latex_num(pb, modo)} = {latex_num(pab, modo)}")
    else:
        pab = Fraction(0)
        pasos.append(r"\text{Mutuamente excluyentes} \Rightarrow P(A\cap B) = 0")

    lo, hi = max(Fraction(0), pa + pb - 1), min(pa, pb)
    if not (lo <= pab <= hi):
        st.error(f"Los datos son inconsistentes: P(A∩B) = {fmt_p(pab, modo)} debe estar entre {fmt_p(lo, modo)} y {fmt_p(hi, modo)}.")
        return

    union = pa + pb - pab
    res = [
        ("P(A ∩ B)", pab, r"P(A\cap B)"),
        ("P(A ∪ B)", union, rf"P(A)+P(B)-P(A\cap B) = {latex_num(pa, modo)}+{latex_num(pb, modo)}-{latex_num(pab, modo)}"),
        ("P(A')", 1 - pa, rf"1-P(A) = 1-{latex_num(pa, modo)}"),
        ("P(B')", 1 - pb, rf"1-P(B) = 1-{latex_num(pb, modo)}"),
        ("P(A ∩ B')  solo A", pa - pab, rf"P(A)-P(A\cap B) = {latex_num(pa, modo)}-{latex_num(pab, modo)}"),
        ("P(A' ∩ B)  solo B", pb - pab, rf"P(B)-P(A\cap B) = {latex_num(pb, modo)}-{latex_num(pab, modo)}"),
        ("P(A' ∩ B')  ninguno", 1 - union, rf"1-P(A\cup B) = 1-{latex_num(union, modo)}"),
        ("P(A Δ B)  exactamente uno", union - pab, rf"P(A\cup B)-P(A\cap B)"),
    ]
    if pb > 0:
        res.append(("P(A | B)", pab / pb, rf"\frac{{P(A\cap B)}}{{P(B)}} = \frac{{{latex_num(pab, modo)}}}{{{latex_num(pb, modo)}}}"))
    if pa > 0:
        res.append(("P(B | A)", pab / pa, rf"\frac{{P(A\cap B)}}{{P(A)}} = \frac{{{latex_num(pab, modo)}}}{{{latex_num(pa, modo)}}}"))
    if pb < 1:
        res.append(("P(A | B')", (pa - pab) / (1 - pb), rf"\frac{{P(A\cap B')}}{{P(B')}}"))

    tabla = pd.DataFrame([(n.replace("A", nA).replace("B", nB), fmt_p(v, modo)) for n, v, _ in res], columns=["Probabilidad", "Valor"])

    col_t, col_v = st.columns([1, 1.2])
    with col_t:
        st.markdown("### Resultados")
        st.dataframe(tabla, hide_index=True)
        indep = pab == pa * pb
        excl = pab == 0
        st.markdown(f"- **¿Independientes?** {'✅ Sí' if indep else '❌ No'} → P(A∩B) = {fmt_p(pab, modo)} vs P(A)·P(B) = {fmt_p(pa * pb, modo)}")
        st.markdown(f"- **¿Mutuamente excluyentes?** {'✅ Sí' if excl else '❌ No'} → P(A∩B) {'=' if excl else '≠'} 0")
    with col_v:
        st.markdown("### Diagrama de Venn")
        opciones = {
            "A ∪ B": {(True, False), (False, True), (True, True)},
            "A ∩ B": {(True, True)},
            "Solo A (A − B)": {(True, False)},
            "Solo B (B − A)": {(False, True)},
            "A'": {(False, True), (False, False)},
            "B'": {(True, False), (False, False)},
            "(A ∪ B)'  ninguno": {(False, False)},
            "A Δ B  exactamente uno": {(True, False), (False, True)},
            "Nada": set(),
        }
        sel = st.selectbox("Región a sombrear", list(opciones), key="p2_region")
        regiones = {(True, False): pa - pab, (False, True): pb - pab, (True, True): pab, (False, False): 1 - union}
        dibujar_venn([nA, nB], regiones, opciones[sel], modo)
        st.caption(f"Probabilidad de la región sombreada: **{fmt_p(sum(regiones[r] for r in opciones[sel]), modo)}**")

    with st.expander("📐 Demostración paso a paso", expanded=True):
        for p in pasos:
            st.latex(p)
        for nombre, valor, form in res:
            izq = nombre.split("  ")[0].replace("∩", r"\cap ").replace("∪", r"\cup ").replace("Δ", r"\,\Delta\, ")
            st.latex(rf"{izq} = {form} = {latex_num(valor, modo)}" if form != r"P(A\cap B)" else rf"{izq} = {latex_num(valor, modo)}")

    codigo_r(f"""
PA <- {float(pa)}; PB <- {float(pb)}; PAB <- {float(pab)}
PAuB <- PA + PB - PAB          # P(A ∪ B)
PA_dado_B <- PAB / PB          # P(A | B)
PB_dado_A <- PAB / PA          # P(B | A)
independientes <- isTRUE(all.equal(PAB, PA * PB))
c(union = PAuB, A_dado_B = PA_dado_B, B_dado_A = PB_dado_A, ninguno = 1 - PAuB)
""")
    reporte(f"Con P({nA}) = {fmt_p(pa, modo)} y P({nB}) = {fmt_p(pb, modo)}, la probabilidad de que ocurran ambos eventos es "
            f"{fmt_p(pab, modo)} y la de que ocurra al menos uno es {fmt_p(union, modo)}. "
            f"Los eventos {'son' if pab == pa * pb else 'no son'} independientes"
            + (f"; además, P({nA}|{nB}) = {fmt_p(pab / pb, modo)}." if pb > 0 else "."), "p2", "Probabilidad: dos eventos")


# ------------------------------------------------------------------
# Tres eventos
# ------------------------------------------------------------------

def _tres_eventos(modo):
    st.caption("Ingresa las probabilidades de cada evento y de sus intersecciones (pueden ser 0).")
    c = st.columns(3)
    nombres = [c[i].text_input(f"Nombre evento {i + 1}", n, key=f"p3_n{i}") for i, n in enumerate("ABC")]
    c = st.columns(3)
    with c[0]:
        pa = _leer_p(f"P({nombres[0]})", "0.5", "p3_a")
    with c[1]:
        pb = _leer_p(f"P({nombres[1]})", "0.4", "p3_b")
    with c[2]:
        pc = _leer_p(f"P({nombres[2]})", "0.3", "p3_c")
    c = st.columns(4)
    with c[0]:
        pab = _leer_p("P(A∩B)", "0.2", "p3_ab")
    with c[1]:
        pac = _leer_p("P(A∩C)", "0.1", "p3_ac")
    with c[2]:
        pbc = _leer_p("P(B∩C)", "0.1", "p3_bc")
    with c[3]:
        pabc = _leer_p("P(A∩B∩C)", "0.05", "p3_abc")
    if None in (pa, pb, pc, pab, pac, pbc, pabc):
        return
    reg = {
        (True, True, True): pabc,
        (True, True, False): pab - pabc,
        (True, False, True): pac - pabc,
        (False, True, True): pbc - pabc,
        (True, False, False): pa - pab - pac + pabc,
        (False, True, False): pb - pab - pbc + pabc,
        (False, False, True): pc - pac - pbc + pabc,
    }
    union = pa + pb + pc - pab - pac - pbc + pabc
    reg[(False, False, False)] = 1 - union
    malos = [k for k, v in reg.items() if v < 0]
    if malos:
        st.error("Los datos son inconsistentes: alguna región del diagrama queda con probabilidad negativa.")
        return

    st.latex(r"P(A\cup B\cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)")
    st.latex(rf"= {latex_num(pa, modo)}+{latex_num(pb, modo)}+{latex_num(pc, modo)}-{latex_num(pab, modo)}-{latex_num(pac, modo)}-{latex_num(pbc, modo)}+{latex_num(pabc, modo)} = {latex_num(union, modo)}")

    opciones = {
        "A ∪ B ∪ C": {k for k in reg if any(k)},
        "Exactamente uno": {k for k in reg if sum(k) == 1},
        "Exactamente dos": {k for k in reg if sum(k) == 2},
        "Al menos dos": {k for k in reg if sum(k) >= 2},
        "Los tres": {(True, True, True)},
        "Ninguno": {(False, False, False)},
        "Solo A": {(True, False, False)},
        "Solo B": {(False, True, False)},
        "Solo C": {(False, False, True)},
    }
    col1, col2 = st.columns([1, 1.2])
    with col1:
        filas = [(n, fmt_p(sum(reg[r] for r in s), modo)) for n, s in opciones.items()]
        st.dataframe(pd.DataFrame(filas, columns=["Evento", "Probabilidad"]), hide_index=True)
    with col2:
        sel = st.selectbox("Región a sombrear", list(opciones), key="p3_sel")
        dibujar_venn(nombres, reg, opciones[sel], modo)
    reporte(f"La probabilidad de que ocurra al menos uno de los eventos {', '.join(nombres)} es {fmt_p(union, modo)}, "
            f"y la de que no ocurra ninguno es {fmt_p(1 - union, modo)}. La probabilidad de que ocurra exactamente uno es "
            f"{fmt_p(sum(reg[r] for r in opciones['Exactamente uno']), modo)}.", "p3", "Probabilidad: tres eventos")


# ------------------------------------------------------------------
# Tabla de contingencia
# ------------------------------------------------------------------

def _contingencia(modo):
    c1, c2 = st.columns(2)
    filas_txt = c1.text_input("Categorías de las filas (separadas por coma)", "Hombre, Mujer", key="ct_filas")
    cols_txt = c2.text_input("Categorías de las columnas (separadas por coma)", "Fuma, No fuma", key="ct_cols")
    c1, c2 = st.columns(2)
    var_f = c1.text_input("Nombre variable filas", "Sexo", key="ct_vf")
    var_c = c2.text_input("Nombre variable columnas", "Hábito", key="ct_vc")
    filas = [f.strip() for f in filas_txt.split(",") if f.strip()]
    cols = [c.strip() for c in cols_txt.split(",") if c.strip()]
    if len(filas) < 2 or len(cols) < 2 or len(set(filas)) != len(filas) or len(set(cols)) != len(cols):
        st.warning("Necesitas al menos 2 categorías distintas en filas y en columnas.")
        return
    ejemplo = [[30, 70], [20, 80]]
    base = pd.DataFrame([[ejemplo[i][j] if i < 2 and j < 2 else 10 for j in range(len(cols))] for i in range(len(filas))],
                        index=filas, columns=cols, dtype=float)
    st.markdown("#### Ingresa las frecuencias observadas (edita las celdas)")
    tabla = st.data_editor(base, key=f"ct_editor_{'|'.join(filas)}_{'|'.join(cols)}")
    tabla = tabla.fillna(0).astype(float)
    if (tabla.values < 0).any():
        st.error("No se permiten valores negativos.")
        return
    total = tabla.values.sum()
    if total <= 0:
        st.error("La tabla está vacía.")
        return

    conj = tabla / total
    con_tot = tabla.copy()
    con_tot["Total"] = con_tot.sum(axis=1)
    con_tot.loc["Total"] = con_tot.sum(axis=0)
    conj_tot = con_tot / total

    t1, t2, t3, t4 = st.tabs(["Conjunta y marginal", "Condicional por filas", "Condicional por columnas", "Gráficos"])
    with t1:
        st.markdown("**Frecuencias con totales**")
        st.dataframe(con_tot.style.format("{:g}"))
        st.markdown("**Probabilidades conjuntas P(fila ∩ columna) y marginales (Total)**")
        st.dataframe(conj_tot.map(lambda v: fmt_p(Fraction(v).limit_denominator(1000000), modo)))
    with t2:
        st.markdown(f"**P({var_c} | {var_f})** — cada fila suma 1")
        st.dataframe(tabla.div(tabla.sum(axis=1), axis=0).map(lambda v: fmt_p(v, modo)))
    with t3:
        st.markdown(f"**P({var_f} | {var_c})** — cada columna suma 1")
        st.dataframe(tabla.div(tabla.sum(axis=0), axis=1).map(lambda v: fmt_p(v, modo)))
    with t4:
        pct = st.radio("Barras", ["Frecuencias", "% dentro de cada fila"], horizontal=True, key="ct_pct")
        datos = tabla if pct == "Frecuencias" else tabla.div(tabla.sum(axis=1), axis=0) * 100
        fig = go.Figure([go.Bar(name=c, x=filas, y=datos[c]) for c in cols])
        fig.update_layout(barmode="group" if pct == "Frecuencias" else "stack", title=f"{var_c} según {var_f}",
                          xaxis_title=var_f, legend_title=var_c)
        mostrar_plotly(fig, key="ct_bar")
        fig2 = go.Figure(go.Heatmap(z=conj.values, x=cols, y=filas, colorscale="Blues",
                                    text=[[fmt_p(v, modo) for v in fila] for fila in conj.values], texttemplate="%{text}"))
        fig2.update_layout(title="Probabilidades conjuntas", yaxis_autorange="reversed")
        mostrar_plotly(fig2, key="ct_heat")

    st.markdown("### 🎯 Calcular probabilidades para dos eventos")
    c1, c2 = st.columns(2)
    A = c1.selectbox(f"Evento A ({var_f} =)", filas, key="ct_A")
    B = c2.selectbox(f"Evento B ({var_c} =)", cols, key="ct_B")
    nA, nB, nAB = tabla.loc[A].sum(), tabla[B].sum(), tabla.loc[A, B]
    to_f = (lambda a: Fraction(int(a), int(total))) if all(float(v).is_integer() for v in (nA, nB, nAB, total)) else (lambda a: a / total)
    pA, pB, pAB = to_f(nA), to_f(nB), to_f(nAB)
    pU = pA + pB - pAB
    filas_res = [
        (f"P({A})", pA, rf"\frac{{{nA:g}}}{{{total:g}}}"),
        (f"P({B})", pB, rf"\frac{{{nB:g}}}{{{total:g}}}"),
        (f"P({A} ∩ {B})", pAB, rf"\frac{{{nAB:g}}}{{{total:g}}}"),
        (f"P({A} ∪ {B})", pU, rf"\frac{{{nA:g}+{nB:g}-{nAB:g}}}{{{total:g}}}"),
        (f"P({A} | {B})", pAB / pB if pB else float("nan"), rf"\frac{{{nAB:g}}}{{{nB:g}}}"),
        (f"P({B} | {A})", pAB / pA if pA else float("nan"), rf"\frac{{{nAB:g}}}{{{nA:g}}}"),
    ]
    col1, col2 = st.columns([1, 1.3])
    with col1:
        st.dataframe(pd.DataFrame([(n, fmt_p(v, modo)) for n, v, _ in filas_res], columns=["Probabilidad", "Valor"]), hide_index=True)
    with col2:
        for n, v, f in filas_res:
            st.latex(rf"{tx(n)} = {f} = {latex_num(v, modo)}")
    indep = abs(float(pAB) - float(pA) * float(pB)) < 1e-12
    st.markdown(f"**¿{A} y {B} son independientes?** {'✅ Sí' if indep else '❌ No'}: P(A∩B) = {fmt_p(pAB, modo)} "
                f"{'=' if indep else '≠'} P(A)·P(B) = {fmt_p(pA * pB, modo)}")

    with st.expander("🧪 Prueba de independencia χ² (para toda la tabla)"):
        try:
            from scipy.stats import chi2_contingency
            chi2, pval, gl, esperadas = chi2_contingency(tabla.values, correction=False)
            alpha = st.number_input("α", 0.001, 0.5, 0.05, key="ct_alpha")
            st.latex(r"\chi^2 = \sum \frac{(O_{ij}-E_{ij})^2}{E_{ij}},\qquad E_{ij} = \frac{(\text{total fila } i)(\text{total columna } j)}{n}")
            st.markdown("**Frecuencias esperadas bajo independencia**")
            st.dataframe(pd.DataFrame(esperadas, index=filas, columns=cols).style.format("{:.3f}"))
            c = st.columns(3)
            c[0].metric("χ² calculado", f"{chi2:.4f}")
            c[1].metric("gl", gl)
            c[2].metric("p-valor", f"{pval:.6f}")
            if (esperadas < 5).any():
                st.warning("Hay frecuencias esperadas menores que 5: la aproximación χ² puede no ser confiable.")
            if pval < alpha:
                st.success(f"Se rechaza H₀: hay evidencia de asociación entre {var_f} y {var_c}.")
            else:
                st.info(f"No se rechaza H₀: no hay evidencia suficiente de asociación entre {var_f} y {var_c}.")
        except Exception as e:
            st.error(f"No se pudo calcular: {e}")

    matriz_r = ", ".join(f"{v:g}" for v in tabla.values.flatten())
    codigo_r(f"""
tabla <- matrix(c({matriz_r}), nrow = {len(filas)}, byrow = TRUE,
                dimnames = list({var_f.replace(' ', '_')} = c({', '.join(repr(f) for f in filas)}),
                                {var_c.replace(' ', '_')} = c({', '.join(repr(c) for c in cols)})))
addmargins(tabla)                     # frecuencias con totales
prop.table(tabla)                     # probabilidades conjuntas
addmargins(prop.table(tabla))         # conjuntas + marginales
prop.table(tabla, margin = 1)         # condicional por filas
prop.table(tabla, margin = 2)         # condicional por columnas
chisq.test(tabla, correct = FALSE)    # prueba de independencia
""".replace("'", '"'))
    reporte(f"De un total de {total:g} observaciones, la probabilidad de {A} es {fmt_p(pA, modo)} y la de {B} es {fmt_p(pB, modo)}. "
            f"La probabilidad conjunta P({A} ∩ {B}) es {fmt_p(pAB, modo)}, y dado que ocurre {B}, la probabilidad de {A} es "
            f"{fmt_p(pAB / pB, modo) if pB else 'indefinida'}. Los eventos {'son' if indep else 'no son'} independientes.",
            "ct", "Tabla de contingencia")


# ------------------------------------------------------------------
# Clásica vs frecuentista
# ------------------------------------------------------------------

def _simulacion_frecuentista(modo):
    st.markdown("La **probabilidad clásica** cuenta casos favorables / casos posibles. La **frecuentista** es el límite de la "
                "frecuencia relativa al repetir el experimento muchas veces (Ley de los Grandes Números).")
    exp = st.selectbox("Experimento", ["Lanzar una moneda", "Lanzar un dado", "Lanzar dos dados (suma)", "Evento con probabilidad p"], key="sim_exp")
    rng_seed = st.number_input("Semilla (cámbiala para otra simulación)", 0, 10 ** 6, 2026, key="sim_seed")
    n = st.slider("Número de repeticiones", 10, 20000, 2000, step=10, key="sim_n")
    rng = np.random.default_rng(int(rng_seed))

    if exp == "Lanzar una moneda":
        lado = st.radio("Evento", ["Cara", "Sello"], horizontal=True, key="sim_lado")
        p_teo = Fraction(1, 2)
        resultados = rng.integers(0, 2, n)
        exito = resultados == (0 if lado == "Cara" else 1)
        desc = f"sale {lado}"
        r_code = f"x <- sample(c('Cara','Sello'), {n}, replace = TRUE)\nmean(x == '{lado}')"
    elif exp == "Lanzar un dado":
        caras = st.multiselect("Evento: sale alguna de estas caras", [1, 2, 3, 4, 5, 6], default=[6], key="sim_caras")
        if not caras:
            st.warning("Selecciona al menos una cara.")
            return
        p_teo = Fraction(len(caras), 6)
        resultados = rng.integers(1, 7, n)
        exito = np.isin(resultados, caras)
        desc = f"sale {', '.join(map(str, caras))}"
        r_code = f"x <- sample(1:6, {n}, replace = TRUE)\nmean(x %in% c({', '.join(map(str, caras))}))"
    elif exp == "Lanzar dos dados (suma)":
        c1, c2 = st.columns(2)
        op = c1.selectbox("Condición", ["=", "≤", "≥"], key="sim_op")
        s = c2.number_input("Suma", 2, 12, 7, key="sim_suma")
        sumas = np.add.outer(np.arange(1, 7), np.arange(1, 7)).flatten()
        f = {"=": np.equal, "≤": np.less_equal, "≥": np.greater_equal}[op]
        p_teo = Fraction(int(f(sumas, s).sum()), 36)
        resultados = rng.integers(1, 7, n) + rng.integers(1, 7, n)
        exito = f(resultados, s)
        desc = f"la suma es {op} {s}"
        r_code = f"s <- sample(1:6, {n}, TRUE) + sample(1:6, {n}, TRUE)\nmean(s {'==' if op == '=' else op.replace('≤', '<=').replace('≥', '>=')} {s})"
        with st.expander("Espacio muestral de dos dados (36 resultados)"):
            m = pd.DataFrame(np.add.outer(np.arange(1, 7), np.arange(1, 7)), index=range(1, 7), columns=range(1, 7))
            st.dataframe(m.style.apply(lambda col: [f"background-color: {COLOR_A}55" if f(v, s) else "" for v in col]))
            st.caption(f"Casos favorables: {int(f(sumas, s).sum())} de 36")
    else:
        p_teo = a_fraccion(st.text_input("p", "0.3", key="sim_p"))
        exito = rng.random(n) < float(p_teo)
        desc = "ocurre el evento"
        r_code = f"x <- rbinom({n}, 1, {float(p_teo)})\nmean(x)"

    frec = np.cumsum(exito) / np.arange(1, n + 1)
    c = st.columns(3)
    c[0].metric("Probabilidad clásica", fmt_p(p_teo, modo))
    c[1].metric(f"Frecuencia relativa ({n} rep.)", f"{frec[-1]:.4f}")
    c[2].metric("Diferencia", f"{abs(frec[-1] - float(p_teo)):.4f}")
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, n + 1), y=frec, mode="lines", name="Frecuencia relativa", line_color=COLOR_PRIMARIO)
    fig.add_hline(y=float(p_teo), line_dash="dash", line_color=COLOR_A, annotation_text=f"P clásica = {float(p_teo):.4f}")
    fig.update_layout(title=f"Ley de los grandes números: P({desc})", xaxis_title="Número de repeticiones",
                      yaxis_title="Frecuencia relativa", xaxis_type="log")
    mostrar_plotly(fig, key="sim_lgn")
    codigo_r("set.seed(" + str(int(rng_seed)) + ")\n" + r_code.replace("'", '"')
             + "\n# Frecuencia relativa acumulada (cambia 'exito' por tu condición lógica)\n"
             + "# exito <- x == \"Cara\"\n# plot(cumsum(exito) / seq_along(exito), type = \"l\", log = \"x\")")
