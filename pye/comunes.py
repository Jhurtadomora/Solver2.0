"""Utilidades compartidas por los módulos de Probabilidad y Estadística."""
import ast
import html
import io
import math
import re
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

COLOR_PRIMARIO = "#38bdf8"
COLOR_A = "#ef4444"
COLOR_B = "#6366f1"
COLOR_OK = "#10b981"
COLOR_ACENTO = "#f59e0b"
COLOR_GRIS = "#94a3b8"


# ------------------------------------------------------------------
# Números, fracciones y listas
# ------------------------------------------------------------------

def a_fraccion(valor):
    """Convierte '1/3', '0,25', '30%', 0.2 ... en una Fraction exacta."""
    if isinstance(valor, Fraction):
        return valor
    if isinstance(valor, (bool, np.bool_)):
        raise ValueError("valor no numérico")
    if isinstance(valor, (int, np.integer)):
        return Fraction(int(valor))
    if isinstance(valor, (float, np.floating)):
        if math.isnan(valor) or math.isinf(valor):
            raise ValueError("valor no finito")
        return Fraction(repr(float(valor)))
    s = str(valor).strip().replace(" ", "")
    if s == "" or s.lower() in ("none", "nan"):
        raise ValueError("entrada vacía")
    if s.endswith("%"):
        return a_fraccion(s[:-1]) / 100
    s = s.replace(",", ".")
    if "/" in s:
        num, den = s.split("/", 1)
        den_f = a_fraccion(den)
        if den_f == 0:
            raise ValueError("división por cero")
        return a_fraccion(num) / den_f
    return Fraction(s)


def a_float(valor):
    return float(a_fraccion(valor))


def separar_tokens(txt):
    """Separa por ';' (permitiendo coma decimal) o por comas / espacios / saltos de línea."""
    txt = (txt or "").strip()
    if not txt:
        return []
    if ";" in txt:
        partes = re.split(r"[;\n\t]+", txt)
    else:
        partes = re.split(r"[,\s]+", txt)
    return [p.strip() for p in partes if p.strip()]


def parse_lista(txt):
    return [a_float(p) for p in separar_tokens(txt)]


def fmt_num(x, dec=4):
    x = float(x)
    if abs(x - round(x)) < 1e-12:
        return f"{int(round(x))}"
    return f"{x:.{dec}f}".rstrip("0").rstrip(".")


def fmt_p(p, modo="Decimal", dec=4):
    """Formatea una probabilidad como decimal, fracción o porcentaje."""
    if modo == "Fracción":
        f = p if isinstance(p, Fraction) else Fraction(float(p)).limit_denominator(100000)
        if f.denominator <= 100000:
            return str(f)
    if modo == "Porcentaje":
        return f"{float(p) * 100:.{max(dec - 2, 2)}f}%"
    return f"{float(p):.{dec}f}"


def latex_num(p, modo="Decimal", dec=4):
    if modo == "Fracción" and isinstance(p, Fraction) and p.denominator != 1 and p.denominator <= 100000:
        signo = "-" if p < 0 else ""
        return rf"{signo}\frac{{{abs(p.numerator)}}}{{{p.denominator}}}"
    if modo == "Porcentaje":
        return rf"{float(p) * 100:.{max(dec - 2, 2)}f}\%"
    return fmt_num(p, dec)


def tx(s):
    """Texto seguro para usar dentro de st.latex."""
    s = str(s)
    reemplazos = [("\\", r"\backslash "), ("{", r"\{"), ("}", r"\}"), ("_", r"\_"),
                  ("^", r"\^{}"), ("#", r"\#"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("~", " ")]
    for a, b in reemplazos:
        s = s.replace(a, b)
    return r"\text{" + s + "}"


# ------------------------------------------------------------------
# Expresiones matemáticas seguras (f(x), g(x), condiciones en x, y)
# ------------------------------------------------------------------

_FUNCIONES = {
    "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "ln": np.log, "log10": np.log10,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "abs": np.abs,
    "pi": np.pi, "e": np.e, "factorial": lambda n: math.factorial(int(n)),
}

_NODOS_OK = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load, ast.Call,
             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv)
_NODOS_LOGICOS = (ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not, ast.Lt, ast.LtE, ast.Gt,
                  ast.GtE, ast.Eq, ast.NotEq)


def compilar_expr(expr, variables=("x",), logica=False):
    """Compila una expresión matemática del usuario de forma segura (sin eval libre)."""
    expr = (expr or "").strip().replace("^", "**").replace("≤", "<=").replace("≥", ">=")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)  # permite 'x = 2'
    if not expr:
        raise ValueError("Escribe una expresión.")
    arbol = ast.parse(expr, mode="eval")
    permitidos = _NODOS_OK + (_NODOS_LOGICOS if logica else ())
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, permitidos):
            raise ValueError(f"Operación no permitida en la expresión: {type(nodo).__name__}")
        if isinstance(nodo, ast.Name) and nodo.id not in _FUNCIONES and nodo.id not in variables:
            raise ValueError(f"Nombre desconocido '{nodo.id}'. Variables permitidas: {', '.join(variables)}")
        if isinstance(nodo, ast.Call) and not (isinstance(nodo.func, ast.Name) and nodo.func.id in _FUNCIONES):
            raise ValueError("Solo se permiten funciones como sqrt, exp, log, sin, cos, abs.")
        if isinstance(nodo, ast.Constant) and not isinstance(nodo.value, (int, float)):
            raise ValueError("Solo se permiten constantes numéricas.")
    codigo = compile(arbol, "<expr>", "eval")

    def funcion(**valores):
        return eval(codigo, {"__builtins__": {}}, {**_FUNCIONES, **valores})  # noqa: S307 (AST validado)

    return funcion


# ------------------------------------------------------------------
# Salidas comunes
# ------------------------------------------------------------------

def reporte(texto, key, modulo="Cálculo"):
    st.markdown("### 📝 Conclusión para informe")
    st.markdown(
        f"<div class='report-box'>{html.escape(texto).replace(chr(10), '<br>')}</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.download_button("Descargar conclusión (TXT)", data=texto, file_name="conclusion_estadistica.txt",
                       mime="text/plain", key=f"dl_rep_{key}")
    if c2.button("Guardar en historial", key=f"hist_rep_{key}"):
        st.session_state.setdefault("historial", []).append({
            "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modulo": modulo,
            "inputs": {"conclusion": texto},
        })
        st.toast("Guardado en el historial de cálculos")


def codigo_r(codigo, titulo="💻 Código R equivalente (para tus entregas en R/RStudio)"):
    with st.expander(titulo):
        st.code(codigo.strip(), language="r")


def r_vector(valores, max_items=300):
    valores = list(valores)
    if len(valores) > max_items:
        return "datos$variable  # (reemplaza por tu columna)"
    if all(isinstance(v, str) for v in valores):
        return "c(" + ", ".join('"' + v.replace('"', "'") + '"' for v in valores) + ")"
    return "c(" + ", ".join(fmt_num(v, 6) for v in valores) + ")"


def descargas_tabla(df, nombre, key):
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Descargar CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{nombre}.csv", mime="text/csv", key=f"csv_{key}")
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="tabla")
        c2.download_button("⬇️ Descargar Excel", buffer.getvalue(), file_name=f"{nombre}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=f"xlsx_{key}")
    except Exception:
        pass


def mostrar_plotly(fig, key=None, **kwargs):
    fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), hoverlabel=dict(namelength=-1))
    return st.plotly_chart(fig, key=key, **kwargs)


def glosario(filas, titulo="📘 Fórmulas y significado de cada símbolo", expandido=True):
    """filas: lista de (símbolo LaTeX, nombre, qué significa, fórmula LaTeX)."""
    with st.expander(titulo, expanded=expandido):
        md = "| Símbolo | Nombre | ¿Qué significa? | Fórmula |\n|---|---|---|---|\n"
        for simbolo, nombre, significado, formula in filas:
            md += f"| ${simbolo}$ | {nombre} | {significado} | ${formula}$ |\n"
        st.markdown(md)


def diagrama_cajas(datos, nombre, key):
    """Diagrama de cajas y bigotes con media, cuartiles y valores atípicos (regla 1.5·RIC)."""
    import plotly.graph_objects as go

    x = np.asarray(datos, dtype=float)
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    ric = q3 - q1
    li, ls = q1 - 1.5 * ric, q3 + 1.5 * ric
    media = float(x.mean())
    atipicos = np.sort(x[(x < li) | (x > ls)])
    normales = x[(x >= li) & (x <= ls)]
    bigote_inf, bigote_sup = (normales.min(), normales.max()) if len(normales) else (q1, q3)

    fig = go.Figure()
    fig.add_trace(go.Box(x=x, name=nombre, boxpoints="outliers", boxmean=True, quartilemethod="linear",
                         marker=dict(color=COLOR_A, size=10, symbol="circle-open", line=dict(width=2)),
                         line=dict(color=COLOR_PRIMARIO), fillcolor="rgba(56,189,248,0.25)", orientation="h",
                         hoverinfo="x"))
    fig.add_trace(go.Scatter(x=[media], y=[nombre], mode="markers+text", text=[f"x̄ = {fmt_num(media, 2)}"],
                             textposition="top center", marker=dict(symbol="x", size=14, color=COLOR_ACENTO),
                             name="Media", hovertemplate="Media = %{x:.4f}<extra></extra>"))
    for valor, txt in [(li, "límite inferior"), (ls, "límite superior")]:
        fig.add_vline(x=valor, line_dash="dot", line_color=COLOR_GRIS, annotation_text=f"{txt} {fmt_num(valor, 2)}",
                      annotation_position="top")
    for v in atipicos:
        fig.add_annotation(x=v, y=nombre, text=f"{fmt_num(v, 2)}", showarrow=True, arrowhead=0, ay=-30,
                           font=dict(color=COLOR_A))
    fig.update_layout(title=f"Diagrama de cajas y bigotes de {nombre}", height=350, showlegend=False, xaxis_title=nombre,
                      margin=dict(t=90))
    mostrar_plotly(fig, key=key)

    c = st.columns(3)
    c[0].metric("Q₁ (primer cuartil)", fmt_num(q1, 2))
    c[1].metric("Mediana Q₂", fmt_num(q2, 2))
    c[2].metric("Q₃ (tercer cuartil)", fmt_num(q3, 2))
    c = st.columns(3)
    c[0].metric("Media x̄ (datos originales)", fmt_num(media, 2))
    c[1].metric("RIC = Q₃ − Q₁", fmt_num(ric, 2))
    c[2].metric("Valores atípicos", str(len(atipicos)))
    if len(atipicos):
        st.warning("Valores atípicos: " + ", ".join(fmt_num(v) for v in atipicos))
    else:
        st.success("No hay valores atípicos: todos los datos están entre los límites.")
    st.caption("El diagrama de cajas se construye con los datos originales (sin agrupar). "
               "En el gráfico: la caja va de Q₁ a Q₃, la línea interna es la mediana, la línea punteada dentro de la caja "
               "es la media (también marcada con ✕) y los círculos rojos son los valores atípicos.")
    with st.expander("📐 ¿De dónde sale cada valor del diagrama?", expanded=False):
        st.latex(rf"Q_1 = P_{{25}} = {fmt_num(q1)},\quad Q_2 = Me = {fmt_num(q2)},\quad Q_3 = P_{{75}} = {fmt_num(q3)}")
        st.latex(rf"RIC = Q_3 - Q_1 = {fmt_num(q3)} - {fmt_num(q1)} = {fmt_num(ric)}")
        st.latex(rf"L_{{inf}} = Q_1 - 1.5\cdot RIC = {fmt_num(q1)} - 1.5({fmt_num(ric)}) = {fmt_num(li)}")
        st.latex(rf"L_{{sup}} = Q_3 + 1.5\cdot RIC = {fmt_num(q3)} + 1.5({fmt_num(ric)}) = {fmt_num(ls)}")
        st.latex(rf"\text{{Bigotes: del menor al mayor dato dentro de los límites}} = [{fmt_num(bigote_inf)},\; {fmt_num(bigote_sup)}]")
        st.markdown("Un dato es **atípico** si es menor que el límite inferior o mayor que el límite superior. "
                    "Los cuartiles se calculan con interpolación lineal: posición = p·(n − 1) en los datos ordenados.")
    return {"q1": q1, "q2": q2, "q3": q3, "ric": ric, "li": li, "ls": ls, "media": media, "atipicos": atipicos}


def entrada_texto_num(label, default, key, minimo=None, maximo=None, ayuda=None):
    """Campo de texto que acepta 0,5 / 1/3 / 30% y devuelve float."""
    txt = st.text_input(label, str(default), key=key, help=ayuda)
    try:
        v = a_float(txt)
    except Exception:
        st.error(f"'{label}': valor no válido. Se usa {default}.")
        v = a_float(default)
    if minimo is not None and v < minimo:
        st.error(f"'{label}' debe ser ≥ {minimo}.")
        v = a_float(default)
    if maximo is not None and v > maximo:
        st.error(f"'{label}' debe ser ≤ {maximo}.")
        v = a_float(default)
    return v


def leer_columna(lector, key, permitir_texto=True):
    """Sube un archivo y devuelve (serie, nombre) de la columna elegida."""
    archivo = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"], key=f"{key}_file")
    if archivo is None:
        return None, None, None
    try:
        df = lector(archivo.getvalue(), archivo.name.lower())
    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return None, None, None
    with st.expander("Vista previa del archivo"):
        st.dataframe(df.head(20))
    columnas = df.columns.tolist() if permitir_texto else df.select_dtypes("number").columns.tolist()
    if not columnas:
        st.error("El archivo no tiene columnas utilizables.")
        return None, None, df
    col = st.selectbox("Variable a analizar", columnas, key=f"{key}_col")
    return df[col], col, df
