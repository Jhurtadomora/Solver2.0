"""CA2 · Árboles de probabilidad interactivos, regla de la multiplicación, probabilidad total y Bayes."""
import hashlib
import itertools
from collections import defaultdict
from fractions import Fraction

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .comunes import (COLOR_A, COLOR_B, COLOR_GRIS, COLOR_OK, COLOR_PRIMARIO, a_fraccion, codigo_r, fmt_num, fmt_p,
                      latex_num, mostrar_plotly, reporte, tx)

MAX_HOJAS_CALCULO = 4096
MAX_HOJAS_DIBUJO = 120


def _hash(*partes):
    return hashlib.md5("§".join(map(str, partes)).encode()).hexdigest()[:10]


def _etiquetas(txt):
    return [e.strip() for e in str(txt).split(",") if e.strip()]


# ------------------------------------------------------------------
# Construcción del árbol
# ------------------------------------------------------------------

def construir_arbol(nombres_etapas, resultados_fn, prob_fn, podar=False):
    arbol = {"etapas": list(nombres_etapas), "rama": {}, "acum": {(): Fraction(1)}, "hojas": []}
    frontera = [()]
    for s in range(len(nombres_etapas)):
        nueva = []
        for camino in frontera:
            for etiqueta in resultados_fn(s, camino):
                p = prob_fn(s, camino, etiqueta)
                if podar and p == 0:
                    continue
                hijo = camino + (etiqueta,)
                arbol["rama"][hijo] = p
                arbol["acum"][hijo] = arbol["acum"][camino] * p
                nueva.append(hijo)
        frontera = nueva
        if len(frontera) > MAX_HOJAS_CALCULO:
            raise ValueError(f"El árbol tendría más de {MAX_HOJAS_CALCULO} caminos. Reduce etapas o resultados.")
    arbol["hojas"] = frontera
    return arbol


def _editor_probabilidades(filas, key, ayuda):
    df = pd.DataFrame(filas, columns=["Viene de", "Resultado", "Probabilidad"])
    editado = st.data_editor(
        df, hide_index=True, disabled=["Viene de", "Resultado"], key=key,
        column_config={"Probabilidad": st.column_config.TextColumn("Probabilidad (condicional)", help=ayuda)},
    )
    return editado


def _validar_nodos(probs, auto_completar):
    """probs: dict padre -> list[(etiqueta, texto)]. Devuelve dict (padre, etiqueta)->Fraction y errores."""
    salida, errores = {}, []
    for padre, items in probs.items():
        valores = []
        for i, (etq, txt) in enumerate(items):
            if auto_completar and i == len(items) - 1:
                valores.append(None)
                continue
            try:
                v = a_fraccion(txt)
                if not 0 <= v <= 1:
                    raise ValueError
                valores.append(v)
            except Exception:
                errores.append(f"Valor no válido '{txt}' en la rama {etq} (desde {' → '.join(padre) or 'Inicio'}).")
                valores.append(Fraction(0))
        if auto_completar:
            valores[-1] = 1 - sum(valores[:-1])
            if valores[-1] < 0:
                errores.append(f"Desde {' → '.join(padre) or 'Inicio'} las probabilidades suman más de 1.")
        suma = sum(valores)
        if suma != 1:
            errores.append(f"Desde **{' → '.join(padre) or 'Inicio'}** las ramas suman {fmt_num(float(suma), 6)} (deben sumar 1).")
        for (etq, _), v in zip(items, valores):
            salida[(padre, etq)] = v
    return salida, errores


def _modo_general():
    S = int(st.number_input("Número de etapas del experimento", 1, 4, 2, key="arb_g_S"))
    ejemplo = [("Máquina", "M1, M2, M3"), ("Calidad", "Defectuoso, Bueno"), ("Etapa 3", "C1, C2"), ("Etapa 4", "D1, D2")]
    nombres, resultados = [], []
    cols = st.columns(S)
    for s in range(S):
        with cols[s]:
            nombres.append(st.text_input(f"Nombre etapa {s + 1}", ejemplo[s][0], key=f"arb_g_n{s}"))
            labs = _etiquetas(st.text_input(f"Resultados etapa {s + 1} (separados por coma)", ejemplo[s][1], key=f"arb_g_r{s}"))
            if len(labs) < 1 or len(set(labs)) != len(labs):
                st.error("Escribe resultados distintos separados por coma.")
                return None, None
            resultados.append(labs)
    total_hojas = int(np.prod([len(r) for r in resultados]))
    if total_hojas > MAX_HOJAS_CALCULO:
        st.error("Demasiados caminos. Reduce etapas o resultados.")
        return None, None

    auto = st.checkbox("Completar automáticamente la última rama de cada nodo (1 − suma de las demás)", True, key="arb_g_auto")
    ejemplo_p = {
        (): ["0.5", "0.3", "0.2"],
        ("M1",): ["0.03", "0.97"], ("M2",): ["0.04", "0.96"], ("M3",): ["0.05", "0.95"],
    }
    probs_txt = {}
    for s in range(S):
        padres = list(itertools.product(*resultados[:s]))
        filas = []
        for padre in padres:
            defecto = ejemplo_p.get(padre) if resultados[0] == ["M1", "M2", "M3"] and s < 2 else None
            if defecto is None or len(defecto) != len(resultados[s]):
                defecto = [f"1/{len(resultados[s])}"] * len(resultados[s])
            for etq, d in zip(resultados[s], defecto):
                filas.append((" → ".join(padre) or "Inicio", etq, d))
        titulo = f"Probabilidades de la etapa {s + 1} · {nombres[s]}" + (" — P(resultado | camino previo)" if s else "")
        with st.expander(titulo, expanded=s < 2):
            editado = _editor_probabilidades(filas, f"arb_g_ed_{s}_{_hash(resultados[: s + 1])}",
                                             "Acepta 0.3, 30%, 3/10. Es la probabilidad condicional dado el camino previo.")
        idx = 0
        for padre in padres:
            probs_txt[padre] = []
            for etq in resultados[s]:
                probs_txt[padre].append((etq, editado.iloc[idx]["Probabilidad"]))
                idx += 1
    probs, errores = _validar_nodos(probs_txt, auto)
    arbol = construir_arbol(nombres, lambda s, c: resultados[s], lambda s, c, e: probs[(c, e)])
    arbol["errores"] = errores
    defaults = {"A": {0: [resultados[0][0]]}, "B": {1: [resultados[1][0]]} if S > 1 else {0: resultados[0][-1:]}}
    return arbol, defaults


def _modo_repetido():
    c1, c2, c3 = st.columns(3)
    n = int(c1.number_input("Número de repeticiones (ensayos)", 1, 8, 3, key="arb_r_n"))
    nombre = c2.text_input("Nombre del ensayo", "Lanzamiento", key="arb_r_nom")
    labs = _etiquetas(c3.text_input("Resultados posibles", "Cara, Sello", key="arb_r_labs"))
    if len(labs) < 1 or len(set(labs)) != len(labs):
        st.error("Escribe resultados distintos separados por coma.")
        return None, None
    if len(labs) ** n > MAX_HOJAS_CALCULO:
        st.error("Demasiados caminos. Reduce repeticiones o resultados.")
        return None, None
    st.caption("Las probabilidades son las mismas en cada ensayo (ensayos independientes, p. ej. con reemplazo).")
    auto = st.checkbox("Completar automáticamente el último resultado", True, key="arb_r_auto")
    editado = _editor_probabilidades([("Cada ensayo", e, f"1/{len(labs)}") for e in labs], f"arb_r_ed_{_hash(labs)}",
                                     "Probabilidad de cada resultado en un ensayo")
    probs, errores = _validar_nodos({(): list(zip(labs, editado["Probabilidad"]))}, auto)
    arbol = construir_arbol([f"{nombre} {i + 1}" for i in range(n)], lambda s, c: labs, lambda s, c, e: probs[((), e)])
    arbol["errores"] = errores
    return arbol, {"A": {"conteo": (labs[0], "Exactamente", min(2, n))}, "B": {0: [labs[0]]}}


def _modo_urna():
    c1, c2 = st.columns([1.3, 1])
    with c1:
        base = pd.DataFrame({"Categoría": ["Roja", "Azul", "Verde"], "Cantidad": [5, 3, 2]})
        tabla = st.data_editor(base, num_rows="dynamic", hide_index=True, key="arb_u_ed",
                               column_config={"Cantidad": st.column_config.NumberColumn(min_value=0, step=1)})
    with c2:
        k = int(st.number_input("Número de extracciones", 1, 5, 2, key="arb_u_k"))
        reemplazo = st.radio("Tipo de extracción", ["Sin reemplazo", "Con reemplazo"], key="arb_u_rep") == "Con reemplazo"
        objeto = st.text_input("¿Qué se extrae?", "Bola", key="arb_u_obj")
    tabla = tabla.dropna()
    tabla = tabla[tabla["Categoría"].astype(str).str.strip() != ""]
    cats = [str(c).strip() for c in tabla["Categoría"]]
    cant = [int(v) for v in tabla["Cantidad"]]
    if len(cats) < 1 or len(set(cats)) != len(cats) or sum(cant) <= 0:
        st.error("Ingresa categorías distintas con cantidades positivas.")
        return None, None
    total = sum(cant)
    if not reemplazo and k > total:
        st.error(f"No se pueden hacer {k} extracciones sin reemplazo de {total} objetos.")
        return None, None
    st.caption(f"Urna con {total} objetos: " + ", ".join(f"{c} = {n}" for c, n in zip(cats, cant)))

    def prob(s, camino, e):
        disponibles = dict(zip(cats, cant))
        if not reemplazo:
            for x in camino:
                disponibles[x] -= 1
        tot = sum(disponibles.values())
        return Fraction(disponibles[e], tot) if tot else Fraction(0)

    arbol = construir_arbol([f"{objeto} {i + 1}" for i in range(k)], lambda s, c: cats, prob, podar=True)
    arbol["errores"] = []
    return arbol, {"A": {"conteo": (cats[0], "Exactamente", 1)}, "B": {0: [cats[0]]}}


def _modo_bayes():
    st.caption("Escribe las hipótesis (causas) con su probabilidad a priori P(Hᵢ) y la probabilidad de la evidencia dada cada hipótesis P(E | Hᵢ).")
    c1, c2 = st.columns([2, 1])
    nom_e = c2.text_input("Nombre de la evidencia E", "Positivo", key="arb_b_e")
    nom_h = c2.text_input("Nombre de la variable hipótesis", "Estado", key="arb_b_h")
    with c1:
        base = pd.DataFrame({"Hipótesis Hᵢ": ["Enfermo", "Sano"], "P(Hᵢ) a priori": ["0.01", "0.99"],
                             "P(E | Hᵢ)": ["0.95", "0.08"]})
        tabla = st.data_editor(base, num_rows="dynamic", hide_index=True, key="arb_b_ed")
    tabla = tabla.dropna()
    hip = [str(h).strip() for h in tabla["Hipótesis Hᵢ"] if str(h).strip()]
    if len(hip) < 2 or len(set(hip)) != len(hip):
        st.error("Necesitas al menos 2 hipótesis con nombres distintos.")
        return None, None
    errores = []
    try:
        pri = [a_fraccion(v) for v in tabla["P(Hᵢ) a priori"]]
        ver = [a_fraccion(v) for v in tabla["P(E | Hᵢ)"]]
    except Exception:
        st.error("Revisa los valores: usa números como 0.3, 30% o 3/10.")
        return None, None
    if any(not 0 <= v <= 1 for v in pri + ver):
        errores.append("Todas las probabilidades deben estar entre 0 y 1.")
    if sum(pri) != 1:
        errores.append(f"Las probabilidades a priori suman {float(sum(pri)):.6f}; deben sumar 1.")
    no_e = f"No {nom_e}"
    probs = {((), h): p for h, p in zip(hip, pri)}
    for h, v in zip(hip, ver):
        probs[((h,), nom_e)] = v
        probs[((h,), no_e)] = 1 - v
    arbol = construir_arbol([nom_h, "Evidencia"], lambda s, c: hip if s == 0 else [nom_e, no_e], lambda s, c, e: probs[(c, e)])
    arbol["errores"] = errores
    return arbol, {"A": {0: [hip[0]]}, "B": {1: [nom_e]}}


# ------------------------------------------------------------------
# Eventos
# ------------------------------------------------------------------

def _resultados_etapa(arbol, s):
    vistos = []
    for h in arbol["hojas"]:
        if h[s] not in vistos:
            vistos.append(h[s])
    return vistos


def _definir_evento(arbol, key, nombre, defecto, chart_key):
    hojas = arbol["hojas"]
    S = len(arbol["etapas"])
    opciones = ["Por resultado en cada etapa", "Por número de veces que aparece un resultado", "Elegir caminos de la lista",
                "Clic / lazo sobre las hojas del gráfico"]
    idx_def = 1 if "conteo" in defecto else 0
    modo = st.selectbox(f"Cómo definir el evento {nombre}", opciones, index=idx_def, key=f"{key}_modo")
    seleccion = set()
    desc = ""
    if modo == opciones[0]:
        partes = []
        sel_total = set(range(len(hojas)))
        for s in range(S):
            res = _resultados_etapa(arbol, s)
            dflt = [r for r in defecto.get(s, res) if r in res] if isinstance(defecto.get(s, res), list) else res
            elegidos = st.multiselect(f"{arbol['etapas'][s]}", res, default=dflt, key=f"{key}_s{s}_{_hash(res)}")
            sel_total &= {i for i, h in enumerate(hojas) if h[s] in elegidos}
            if set(elegidos) != set(res):
                partes.append(f"{arbol['etapas'][s]} ∈ {{{', '.join(elegidos)}}}" if len(elegidos) != 1 else f"{arbol['etapas'][s]} = {elegidos[0]}")
        seleccion = sel_total
        desc = " y ".join(partes) or "cualquier resultado (todo el espacio muestral)"
    elif modo == opciones[1]:
        todas = []
        for s in range(S):
            for r in _resultados_etapa(arbol, s):
                if r not in todas:
                    todas.append(r)
        etq_def, comp_def, k_def = defecto.get("conteo", (todas[0], "Exactamente", 1))
        c1, c2, c3 = st.columns(3)
        etq = c1.selectbox("Resultado", todas, index=todas.index(etq_def) if etq_def in todas else 0, key=f"{key}_ce")
        comp = c2.selectbox("Cuántas veces", ["Exactamente", "Al menos", "A lo sumo"], index=["Exactamente", "Al menos", "A lo sumo"].index(comp_def), key=f"{key}_cc")
        k = int(c3.number_input("k", 0, S, min(k_def, S), key=f"{key}_ck"))
        f = {"Exactamente": lambda c: c == k, "Al menos": lambda c: c >= k, "A lo sumo": lambda c: c <= k}[comp]
        seleccion = {i for i, h in enumerate(hojas) if f(sum(1 for x in h if x == etq))}
        desc = f"{comp.lower()} {k} vez/veces '{etq}'"
    elif modo == opciones[2]:
        etiquetas = [" → ".join(h) for h in hojas]
        elegidos = st.multiselect("Caminos que forman el evento", etiquetas, default=etiquetas[:1], key=f"{key}_lista")
        seleccion = {etiquetas.index(e) for e in elegidos}
        desc = "{" + "; ".join(elegidos) + "}"
    else:
        estado = st.session_state.get(chart_key)
        puntos = []
        try:
            puntos = estado["selection"]["points"] if estado else []
        except Exception:
            puntos = []
        for p in puntos:
            cd = p.get("customdata")
            if cd is not None:
                seleccion.add(int(cd[0] if isinstance(cd, (list, tuple)) else cd))
        st.caption("Haz clic sobre las hojas del árbol (Shift + clic para varias, o usa la herramienta lazo). "
                   f"Seleccionadas: {len(seleccion)}")
        desc = "{" + "; ".join(" → ".join(hojas[i]) for i in sorted(seleccion)) + "}"
    if st.checkbox(f"Usar el complemento de {nombre} ({nombre}')", key=f"{key}_comp"):
        seleccion = set(range(len(hojas))) - seleccion
        desc = f"NO ({desc})"
    return seleccion, desc


# ------------------------------------------------------------------
# Dibujo interactivo
# ------------------------------------------------------------------

def dibujar_arbol(arbol, modo_fmt, resaltadas=frozenset(), vertical=False, key="arbol", mostrar_acum=True,
                  seleccionable=False, color_res=COLOR_A, etiqueta_raiz="Inicio"):
    hojas = arbol["hojas"]
    S = len(arbol["etapas"])
    if len(hojas) > MAX_HOJAS_DIBUJO:
        st.info(f"El árbol tiene {len(hojas)} caminos; se omite el dibujo (máximo {MAX_HOJAS_DIBUJO}). Los cálculos siguen disponibles.")
        return None
    hijos = defaultdict(list)
    for camino in arbol["rama"]:
        hijos[camino[:-1]].append(camino)
    pos = {h: -i for i, h in enumerate(hojas)}

    def ubicar(c):
        if c in pos:
            return pos[c]
        ys = [ubicar(h) for h in hijos[c]]
        pos[c] = sum(ys) / len(ys) if ys else 0
        return pos[c]

    ubicar(())
    resaltados_prefijos = set()
    for i in resaltadas:
        h = hojas[i]
        for k in range(len(h) + 1):
            resaltados_prefijos.add(h[:k])

    def xy(c):
        return (pos[c] * 1.0, -len(c) * 1.0) if vertical else (len(c) * 1.0, pos[c])

    fig = go.Figure()
    for grupo, color, ancho in [(False, COLOR_GRIS, 1.6), (True, color_res, 3.5)]:
        ex, ey = [], []
        for c in arbol["rama"]:
            if (c in resaltados_prefijos) != grupo:
                continue
            x0, y0 = xy(c[:-1])
            x1, y1 = xy(c)
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        if ex:
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color=color, width=ancho), hoverinfo="skip"))

    # probabilidades sobre las ramas
    lx, ly, lt, lh = [], [], [], []
    for c, p in arbol["rama"].items():
        x0, y0 = xy(c[:-1])
        x1, y1 = xy(c)
        lx.append(x0 + (x1 - x0) * 0.55)
        ly.append(y0 + (y1 - y0) * 0.55)
        lt.append(f"<b>{fmt_p(p, modo_fmt)}</b>")
        condicion = " ∩ ".join(c[:-1])
        lh.append(f"P({c[-1]}{' | ' + condicion if condicion else ''}) = {fmt_p(p, modo_fmt)}")
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=lt, textposition="top center" if not vertical else "middle right",
                             textfont=dict(size=12, color=COLOR_PRIMARIO), hovertext=lh, hoverinfo="text"))

    # nodos internos
    internos = [c for c in arbol["acum"] if c not in set(hojas)]
    nx, ny = zip(*[xy(c) for c in internos])
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers+text", text=[c[-1] if c else etiqueta_raiz for c in internos],
        textposition="top center", marker=dict(size=11, color=[color_res if c in resaltados_prefijos and c else "#64748b" for c in internos]),
        hovertext=[f"{' ∩ '.join(c) or etiqueta_raiz}<br>P acumulada = {fmt_p(arbol['acum'][c], modo_fmt)}" for c in internos],
        hoverinfo="text"))

    # hojas
    hx, hy = zip(*[xy(h) for h in hojas])
    textos = []
    for h in hojas:
        t = f"{h[-1]}"
        if mostrar_acum:
            t += f"   <b>P = {fmt_p(arbol['acum'][h], modo_fmt)}</b>"
        textos.append(t)
    hover = []
    for h in hojas:
        factores = " × ".join(fmt_p(arbol["rama"][h[:k + 1]], modo_fmt) for k in range(len(h)))
        hover.append(f"<b>{' → '.join(h)}</b><br>P = {factores} = {fmt_p(arbol['acum'][h], modo_fmt)}")
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers+text", text=textos, textposition="middle right" if not vertical else "bottom center",
        marker=dict(size=14, symbol="circle", color=[color_res if i in resaltadas else COLOR_OK for i in range(len(hojas))],
                    line=dict(width=1, color="white")),
        customdata=[[i] for i in range(len(hojas))], hovertext=hover, hoverinfo="text"))

    # títulos de etapas
    for s, nom in enumerate([etiqueta_raiz] + arbol["etapas"]):
        if s == 0:
            continue
        if vertical:
            fig.add_annotation(x=min(pos.values()) - 0.8, y=-s, text=f"<i>{nom}</i>", showarrow=False, xanchor="right")
        else:
            fig.add_annotation(x=s, y=0.9, text=f"<i>{nom}</i>", showarrow=False, yanchor="bottom")

    n = len(hojas)
    if vertical:
        alto = 180 + 150 * S
        fig.update_xaxes(range=[min(pos.values()) - 2.5, 1.0])
    else:
        alto = max(360, 42 * n + 110)
        fig.update_xaxes(range=[-0.25, S + (1.35 if mostrar_acum else 0.6)])
        fig.update_yaxes(range=[-n + 0.3, 1.6])
    fig.update_layout(showlegend=False, height=alto, clickmode="event+select", dragmode="pan" if not seleccionable else "select",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)")
    if seleccionable:
        return mostrar_plotly(fig, key=key, on_select="rerun", selection_mode=("points", "box", "lasso"))
    return mostrar_plotly(fig, key=key)


# ------------------------------------------------------------------
# Página principal
# ------------------------------------------------------------------

def render_arboles():
    modo = st.radio("Tipo de experimento", [
        "🌳 Constructor general (probabilidades condicionales)",
        "🔁 Ensayos repetidos (monedas, dados, aciertos…)",
        "🎱 Urna: extracciones con o sin reemplazo",
        "🧠 Teorema de Bayes (hipótesis y evidencia)",
    ], key="arb_modo")
    try:
        if modo.startswith("🌳"):
            arbol, defaults = _modo_general()
        elif modo.startswith("🔁"):
            arbol, defaults = _modo_repetido()
        elif modo.startswith("🎱"):
            arbol, defaults = _modo_urna()
        else:
            arbol, defaults = _modo_bayes()
    except ValueError as e:
        st.error(str(e))
        return
    if arbol is None:
        return
    hojas = arbol["hojas"]
    errores = arbol.get("errores", [])
    clave = "arb_" + _hash(modo, [list(h) for h in hojas])
    chart_key = f"{clave}_chart"

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    fmt = c1.radio("Formato de probabilidades", ["Fracción", "Decimal", "Porcentaje"], horizontal=True, key="arb_fmt",
                   index=0 if modo.startswith("🎱") else 1)
    vertical = c2.toggle("Árbol vertical", key="arb_vert")
    mostrar_acum = c3.toggle("Mostrar P de cada camino", True, key="arb_acum")

    for e in errores:
        st.error(e)

    st.markdown("### 🎯 Eventos")
    cA, cB = st.columns(2)
    with cA:
        st.markdown(f"<span style='color:{COLOR_A};font-weight:700'>Evento A</span>", unsafe_allow_html=True)
        A, descA = _definir_evento(arbol, f"{clave}_A", "A", defaults["A"], chart_key)
    with cB:
        st.markdown(f"<span style='color:{COLOR_B};font-weight:700'>Evento B</span>", unsafe_allow_html=True)
        B, descB = _definir_evento(arbol, f"{clave}_B", "B", defaults["B"], chart_key)

    ops = {"A": A, "B": B, "A ∩ B": A & B, "A ∪ B": A | B, "Ninguno": set()}
    resaltar = st.radio("Resaltar en el árbol", list(ops), horizontal=True, key="arb_res")
    color = COLOR_B if resaltar == "B" else COLOR_A

    st.markdown("### 🌳 Diagrama de árbol")
    st.caption("Pasa el mouse sobre ramas y hojas para ver el cálculo. Usa la rueda para hacer zoom y arrastra para moverte.")
    dibujar_arbol(arbol, fmt, ops[resaltar], vertical, key=chart_key, mostrar_acum=mostrar_acum, seleccionable=True, color_res=color)

    if errores:
        st.warning("Corrige las probabilidades marcadas en rojo para ver los cálculos.")
        return

    acum = arbol["acum"]
    pA = sum((acum[hojas[i]] for i in A), Fraction(0))
    pB = sum((acum[hojas[i]] for i in B), Fraction(0))
    pAB = sum((acum[hojas[i]] for i in A & B), Fraction(0))
    pAuB = pA + pB - pAB

    c = st.columns(5)
    c[0].metric("P(A)", fmt_p(pA, fmt))
    c[1].metric("P(B)", fmt_p(pB, fmt))
    c[2].metric("P(A ∩ B)", fmt_p(pAB, fmt))
    c[3].metric("P(A | B)", fmt_p(pAB / pB, fmt) if pB else "—")
    c[4].metric("P(B | A)", fmt_p(pAB / pA, fmt) if pA else "—")
    st.caption(f"**A** = {descA}  ·  **B** = {descB}")

    tabs = st.tabs(["📋 Espacio muestral", "📐 Demostración", "🔁 Bayes", "🎲 Variable aleatoria", "🧪 Simulación", "💻 Código R"])
    with tabs[0]:
        _tab_espacio(arbol, A, B, fmt, clave)
    with tabs[1]:
        _tab_demostracion(arbol, A, B, pA, pB, pAB, fmt, descA, descB)
    with tabs[2]:
        _tab_bayes(arbol, B, pB, fmt, descB, clave)
    with tabs[3]:
        _tab_variable(arbol, fmt, clave)
    with tabs[4]:
        _tab_simulacion(arbol, A, pA, clave)
    with tabs[5]:
        _tab_r(arbol, A, B)

    reporte(f"El experimento tiene {len(hojas)} resultados posibles (caminos del árbol). "
            f"La probabilidad del evento A ({descA}) es {fmt_p(pA, fmt)} y la del evento B ({descB}) es {fmt_p(pB, fmt)}. "
            f"La probabilidad de que ocurran ambos es {fmt_p(pAB, fmt)} y la de que ocurra al menos uno es {fmt_p(pAuB, fmt)}. "
            + (f"Sabiendo que ocurrió B, la probabilidad de A se actualiza a {fmt_p(pAB / pB, fmt)}. " if pB else "")
            + f"Los eventos {'son' if pAB == pA * pB else 'no son'} independientes.", clave, "Árbol de probabilidad")


def _tab_espacio(arbol, A, B, fmt, clave):
    hojas = arbol["hojas"]
    filas = []
    for i, h in enumerate(hojas):
        fila = {"#": i + 1}
        for s, nom in enumerate(arbol["etapas"]):
            fila[nom] = h[s]
        fila["Cálculo"] = " × ".join(fmt_p(arbol["rama"][h[:k + 1]], fmt) for k in range(len(h)))
        fila["P(camino)"] = fmt_p(arbol["acum"][h], fmt)
        fila["∈ A"] = "✅" if i in A else ""
        fila["∈ B"] = "✅" if i in B else ""
        filas.append(fila)
    df = pd.DataFrame(filas)
    st.dataframe(df, hide_index=True)
    total = sum(arbol["acum"][h] for h in hojas)
    st.markdown(f"**Suma de todas las probabilidades de los caminos:** {fmt_p(total, fmt)} {'✅' if total == 1 else '⚠️'}")
    st.download_button("⬇️ Descargar espacio muestral (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="espacio_muestral.csv", key=f"{clave}_csv")


def _factores_latex(arbol, h, fmt):
    nombres = [tx(x) for x in h]
    simb, nums = [], []
    for k in range(len(h)):
        cond = r" \cap ".join(nombres[:k])
        sep = r" \mid " + cond if cond else ""
        simb.append(rf"P({nombres[k]}{sep})")
        nums.append(latex_num(arbol["rama"][h[:k + 1]], fmt))
    inter = r" \cap ".join(nombres)
    return rf"P({inter}) = " + r"\,".join(simb) + " = " + r" \cdot ".join(nums) + f" = {latex_num(arbol['acum'][h], fmt)}"


def _suma_latex(etiqueta, arbol, indices, fmt, max_terminos=10):
    hojas = arbol["hojas"]
    vals = [arbol["acum"][hojas[i]] for i in sorted(indices)]
    total = sum(vals, Fraction(0))
    if not vals:
        return rf"{etiqueta} = 0"
    terminos = " + ".join(latex_num(v, fmt) for v in vals[:max_terminos])
    if len(vals) > max_terminos:
        terminos += rf" + \cdots \;({len(vals)}\text{{ términos}})"
    return rf"{etiqueta} = {terminos} = {latex_num(total, fmt)}"


def _tab_demostracion(arbol, A, B, pA, pB, pAB, fmt, descA, descB):
    hojas = arbol["hojas"]
    st.markdown("#### 1) Regla de la multiplicación: probabilidad de cada camino")
    st.latex(r"P(E_1\cap E_2\cap\cdots\cap E_k) = P(E_1)\,P(E_2\mid E_1)\cdots P(E_k\mid E_1\cap\cdots\cap E_{k-1})")
    limite = st.slider("Caminos a mostrar", 1, len(hojas), min(len(hojas), 8), key="arb_dem_lim") if len(hojas) > 1 else 1
    for h in hojas[:limite]:
        st.latex(_factores_latex(arbol, h, fmt))

    st.markdown("#### 2) Verificación: en cada nodo las ramas suman 1 y todos los caminos suman 1")
    hijos = defaultdict(list)
    for c in arbol["rama"]:
        hijos[c[:-1]].append(c)
    nodos_ok = all(sum(arbol["rama"][h] for h in hs) == 1 for hs in hijos.values())
    st.markdown(f"- Nodos revisados: {len(hijos)} → {'✅ todos suman 1' if nodos_ok else '⚠️ hay nodos que no suman 1'}")
    st.latex(_suma_latex(r"\sum P(\text{caminos})", arbol, range(len(hojas)), fmt, 12))

    st.markdown("#### 3) Probabilidad de un evento = suma de sus caminos (probabilidad total)")
    st.latex(_suma_latex("P(A)", arbol, A, fmt))
    st.latex(_suma_latex("P(B)", arbol, B, fmt))
    st.latex(_suma_latex(r"P(A\cap B)", arbol, A & B, fmt))
    st.latex(rf"P(A\cup B) = P(A)+P(B)-P(A\cap B) = {latex_num(pA, fmt)} + {latex_num(pB, fmt)} - {latex_num(pAB, fmt)} = {latex_num(pA + pB - pAB, fmt)}")
    st.latex(rf"P(A') = 1 - P(A) = {latex_num(1 - pA, fmt)}")

    st.markdown("#### 4) Probabilidad condicional")
    if pB:
        st.latex(rf"P(A\mid B) = \frac{{P(A\cap B)}}{{P(B)}} = \frac{{{latex_num(pAB, fmt)}}}{{{latex_num(pB, fmt)}}} = {latex_num(pAB / pB, fmt)}")
    if pA:
        st.latex(rf"P(B\mid A) = \frac{{P(A\cap B)}}{{P(A)}} = \frac{{{latex_num(pAB, fmt)}}}{{{latex_num(pA, fmt)}}} = {latex_num(pAB / pA, fmt)}")

    st.markdown("#### 5) ¿Son independientes?")
    indep = pAB == pA * pB
    rel = "=" if indep else r"\neq"
    st.latex(rf"P(A)\,P(B) = {latex_num(pA, fmt)}\cdot{latex_num(pB, fmt)} = {latex_num(pA * pB, fmt)} \;{rel}\; P(A\cap B) = {latex_num(pAB, fmt)}")
    st.markdown("✅ **Sí son independientes**." if indep else "❌ **No son independientes**: saber que ocurrió uno cambia la probabilidad del otro.")
    if not (A & B) and A and B:
        st.markdown("Además, A y B son **mutuamente excluyentes** (no comparten caminos).")


def _tab_bayes(arbol, B, pB, fmt, descB, clave):
    hojas = arbol["hojas"]
    S = len(arbol["etapas"])
    if S < 2:
        st.info("El teorema de Bayes requiere al menos 2 etapas (una de hipótesis y otra de evidencia).")
        return
    s = st.selectbox("Etapa que contiene las hipótesis (causas) Hᵢ", range(S), format_func=lambda i: arbol["etapas"][i], key=f"{clave}_bay_s")
    st.markdown(f"**Evidencia observada (evento B):** {descB}")
    if pB == 0:
        st.warning("P(B) = 0: no se puede condicionar en un evento imposible. Cambia la definición de B.")
        return
    hip = _resultados_etapa(arbol, s)
    filas = []
    for h in hip:
        idx = {i for i, x in enumerate(hojas) if x[s] == h}
        pH = sum((arbol["acum"][hojas[i]] for i in idx), Fraction(0))
        pHB = sum((arbol["acum"][hojas[i]] for i in idx & B), Fraction(0))
        filas.append((h, pH, pHB / pH if pH else Fraction(0), pHB, pHB / pB))
    df = pd.DataFrame([(h, fmt_p(a, fmt), fmt_p(b, fmt), fmt_p(c, fmt), fmt_p(d, fmt)) for h, a, b, c, d in filas],
                      columns=["Hipótesis Hᵢ", "P(Hᵢ) a priori", "P(B | Hᵢ) verosimilitud", "P(Hᵢ ∩ B)", "P(Hᵢ | B) a posteriori"])
    st.dataframe(df, hide_index=True)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        elegido = st.selectbox("Demostrar para la hipótesis", hip, key=f"{clave}_bay_h")
        h, pH, pBH, pHB, post = next(f for f in filas if f[0] == elegido)
        st.latex(r"P(H_i\mid B) = \frac{P(H_i)\,P(B\mid H_i)}{\sum_j P(H_j)\,P(B\mid H_j)}")
        denom = " + ".join(rf"{latex_num(a, fmt)}\cdot{latex_num(b, fmt)}" for _, a, b, _, _ in filas[:6])
        if len(filas) > 6:
            denom += r" + \cdots"
        st.latex(rf"P(B) = {denom} = {latex_num(pB, fmt)}")
        st.latex(rf"P({tx(h)}\mid B) = \frac{{{latex_num(pH, fmt)}\cdot{latex_num(pBH, fmt)}}}{{{latex_num(pB, fmt)}}} = \frac{{{latex_num(pHB, fmt)}}}{{{latex_num(pB, fmt)}}} = {latex_num(post, fmt)}")
        cambio = float(post) - float(pH)
        st.info(f"Después de observar B, la probabilidad de **{h}** pasa de {fmt_p(pH, fmt)} a {fmt_p(post, fmt)} "
                f"({'aumenta' if cambio > 0 else 'disminuye' if cambio < 0 else 'no cambia'} {abs(cambio) * 100:.2f} puntos porcentuales).")
    with c2:
        fig = go.Figure()
        fig.add_bar(x=hip, y=[float(f[1]) for f in filas], name="A priori P(Hᵢ)", marker_color=COLOR_GRIS)
        fig.add_bar(x=hip, y=[float(f[4]) for f in filas], name="A posteriori P(Hᵢ | B)", marker_color=COLOR_A)
        fig.update_layout(barmode="group", title="Actualización bayesiana", yaxis_title="Probabilidad", legend=dict(orientation="h"))
        mostrar_plotly(fig, key=f"{clave}_bay_bar")

    st.markdown("#### 🔄 Árbol invertido (primero la evidencia, luego la hipótesis)")
    st.caption("Este árbol tiene las mismas intersecciones que el original, pero ordenadas al revés: así se lee directamente P(Hᵢ | B).")
    pnoB = 1 - pB
    inv = {}
    for h in hip:
        idx = {i for i, x in enumerate(hojas) if x[s] == h}
        pHB = sum((arbol["acum"][hojas[i]] for i in idx & B), Fraction(0))
        pH = sum((arbol["acum"][hojas[i]] for i in idx), Fraction(0))
        inv[("B", h)] = pHB / pB
        inv[("B'", h)] = (pH - pHB) / pnoB if pnoB else Fraction(0)
    arbol_inv = construir_arbol(["Evidencia", arbol["etapas"][s]],
                                lambda k, c: ["B", "B'"] if k == 0 else hip,
                                lambda k, c, e: (pB if e == "B" else pnoB) if k == 0 else inv[(c[0], e)], podar=True)
    dibujar_arbol(arbol_inv, fmt, {i for i, h in enumerate(arbol_inv["hojas"]) if h[0] == "B"}, key=f"{clave}_inv", color_res=COLOR_B)

    codigo_r(f"""
prior <- c({', '.join(fmt_num(float(f[1]), 8) for f in filas)})          # P(H_i)
verosimilitud <- c({', '.join(fmt_num(float(f[2]), 8) for f in filas)})  # P(B | H_i)
PB <- sum(prior * verosimilitud)                                     # probabilidad total
posterior <- prior * verosimilitud / PB                              # teorema de Bayes
names(posterior) <- c({', '.join('"' + h + '"' for h in hip)})
PB; round(posterior, 4)
""")


def _tab_variable(arbol, fmt, clave):
    st.markdown("Asigna un **valor numérico** a cada resultado. La variable X de cada camino es la **suma** de los valores de sus ramas "
                "(p. ej. 1 al resultado que quieres contar y 0 a los demás → X = número de veces que ocurre).")
    etiquetas = []
    for s in range(len(arbol["etapas"])):
        for r in _resultados_etapa(arbol, s):
            if r not in etiquetas:
                etiquetas.append(r)
    base = pd.DataFrame({"Resultado": etiquetas, "Valor": [1.0 if i == 0 else 0.0 for i in range(len(etiquetas))]})
    tabla = st.data_editor(base, hide_index=True, disabled=["Resultado"], key=f"{clave}_va_ed")
    valores = dict(zip(tabla["Resultado"], tabla["Valor"].fillna(0).astype(float)))
    dist = defaultdict(Fraction)
    for h in arbol["hojas"]:
        dist[round(sum(valores[x] for x in h), 10)] += arbol["acum"][h]
    xs = sorted(dist)
    ps = [dist[x] for x in xs]
    E = sum((Fraction(x).limit_denominator(10 ** 9) * p for x, p in zip(xs, ps)), Fraction(0))
    E2 = sum((Fraction(x).limit_denominator(10 ** 9) ** 2 * p for x, p in zip(xs, ps)), Fraction(0))
    V = E2 - E ** 2
    df = pd.DataFrame({"x": [fmt_num(x) for x in xs], "P(X = x)": [fmt_p(p, fmt) for p in ps],
                       "x·P(x)": [fmt_num(x * float(p), 6) for x, p in zip(xs, ps)],
                       "x²·P(x)": [fmt_num(x * x * float(p), 6) for x, p in zip(xs, ps)]})
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(df, hide_index=True)
        m = st.columns(3)
        m[0].metric("E(X)", fmt_p(E, fmt) if fmt == "Fracción" else fmt_num(float(E)))
        m[1].metric("Var(X)", fmt_p(V, fmt) if fmt == "Fracción" else fmt_num(float(V)))
        m[2].metric("σ", fmt_num(float(V) ** 0.5))
    with c2:
        fig = go.Figure(go.Bar(x=xs, y=[float(p) for p in ps], marker_color=COLOR_PRIMARIO, text=[fmt_p(p, fmt) for p in ps], textposition="outside"))
        fig.add_vline(x=float(E), line_dash="dash", line_color=COLOR_A, annotation_text=f"E(X) = {float(E):.4f}")
        fig.update_layout(title="Distribución de probabilidad de X", xaxis_title="x", yaxis_title="P(X = x)", bargap=0.5)
        mostrar_plotly(fig, key=f"{clave}_va_bar")
    terminos = " + ".join(rf"({fmt_num(x)})({latex_num(p, fmt)})" for x, p in list(zip(xs, ps))[:8])
    if len(xs) > 8:
        terminos += r" + \cdots"
    st.latex(rf"E(X) = \sum x\,P(x) = {terminos} = {latex_num(E, fmt)}")
    st.latex(rf"Var(X) = E(X^2) - [E(X)]^2 = {latex_num(E2, fmt)} - ({latex_num(E, fmt)})^2 = {latex_num(V, fmt)}")


def _tab_simulacion(arbol, A, pA, clave):
    hojas = arbol["hojas"]
    c1, c2 = st.columns(2)
    N = c1.slider("Repeticiones del experimento", 100, 50000, 5000, step=100, key=f"{clave}_simN")
    seed = c2.number_input("Semilla", 0, 10 ** 6, 7, key=f"{clave}_seed")
    rng = np.random.default_rng(int(seed))
    probs = np.array([float(arbol["acum"][h]) for h in hojas])
    probs = probs / probs.sum()
    sim = rng.choice(len(hojas), size=N, p=probs)
    en_A = np.isin(sim, list(A))
    frec = np.cumsum(en_A) / np.arange(1, N + 1)
    m = st.columns(3)
    m[0].metric("P(A) teórica", f"{float(pA):.4f}")
    m[1].metric("Frecuencia relativa simulada", f"{frec[-1]:.4f}")
    m[2].metric("Error", f"{abs(frec[-1] - float(pA)):.4f}")
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, N + 1), y=frec, mode="lines", name="Simulación", line_color=COLOR_PRIMARIO)
    fig.add_hline(y=float(pA), line_dash="dash", line_color=COLOR_A, annotation_text="P(A) teórica")
    fig.update_layout(title="Convergencia de la frecuencia relativa de A", xaxis_type="log", xaxis_title="Repeticiones", yaxis_title="Frecuencia")
    mostrar_plotly(fig, key=f"{clave}_sim_fig")
    if len(hojas) <= 40:
        conteo = np.bincount(sim, minlength=len(hojas)) / N
        fig2 = go.Figure()
        etiquetas = [" → ".join(h) for h in hojas]
        fig2.add_bar(x=etiquetas, y=probs, name="Teórica", marker_color=COLOR_GRIS)
        fig2.add_bar(x=etiquetas, y=conteo, name="Simulada", marker_color=COLOR_OK)
        fig2.update_layout(barmode="group", title="Probabilidad de cada camino: teórica vs simulada")
        mostrar_plotly(fig2, key=f"{clave}_sim_bar")


def _tab_r(arbol, A, B):
    hojas = arbol["hojas"]
    caminos = ",\n             ".join('"' + " - ".join(h) + '"' for h in hojas)
    probs = ",\n             ".join(" * ".join(fmt_num(float(arbol["rama"][h[:k + 1]]), 8) for k in range(len(h))) for h in hojas)
    vec = lambda s: "c(" + ", ".join(str(i + 1) for i in sorted(s)) + ")" if s else "integer(0)"
    codigo_r(f"""
# Probabilidad de cada camino = producto de las ramas (regla de la multiplicación)
arbol <- data.frame(
  camino = c({caminos}),
  prob   = c({probs})
)
sum(arbol$prob)                 # debe ser 1
A <- {vec(A)}                   # filas (caminos) que forman A
B <- {vec(B)}                   # filas (caminos) que forman B
PA  <- sum(arbol$prob[A]); PB <- sum(arbol$prob[B])
PAB <- sum(arbol$prob[intersect(A, B)])
PA_dado_B <- PAB / PB
c(PA = PA, PB = PB, PAB = PAB, PA_dado_B = PA_dado_B)

# Simulación del experimento
set.seed(7)
sim <- sample(seq_len(nrow(arbol)), 10000, replace = TRUE, prob = arbol$prob)
mean(sim %in% A)                # frecuencia relativa de A
""", titulo="💻 Código R del árbol")
