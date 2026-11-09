"""CA2 · Técnicas de conteo con listado de arreglos y árbol de posibilidades."""
import itertools
import math
from fractions import Fraction

import pandas as pd
import streamlit as st

from .arboles import construir_arbol, dibujar_arbol
from .comunes import codigo_r, reporte

TEMAS = [
    "Principio multiplicativo",
    "Permutación (todos los elementos): n!",
    "Permutación / variación sin repetición: P(n, r)",
    "Variación con repetición: nʳ",
    "Combinación sin repetición: C(n, r)",
    "Combinación con repetición: CR(n, r)",
    "Permutación con elementos repetidos: n!/(n₁!n₂!…)",
    "Permutación circular: (n − 1)!",
]


def render_conteo():
    tema = st.selectbox("Técnica", TEMAS, key="cnt_tema")
    st.caption("💡 ¿Importa el orden? → permutación/variación. ¿No importa? → combinación. ¿Se pueden repetir? → 'con repetición'.")

    if tema == "Principio multiplicativo":
        txt = st.text_input("Número de opciones en cada etapa (separadas por coma)", "3, 2, 4", key="cnt_mult")
        try:
            etapas = [int(v) for v in txt.split(",") if v.strip()]
            assert etapas and all(e > 0 for e in etapas)
        except Exception:
            st.error("Escribe enteros positivos separados por coma.")
            return
        res = math.prod(etapas)
        st.latex(r"N = n_1\cdot n_2\cdots n_k = " + r"\cdot".join(map(str, etapas)) + f" = {res:,}")
        if res <= 60 and len(etapas) <= 4:
            labs = [[f"{chr(65 + s)}{i + 1}" for i in range(e)] for s, e in enumerate(etapas)]
            arbol = construir_arbol([f"Etapa {s + 1}" for s in range(len(etapas))], lambda s, c: labs[s],
                                    lambda s, c, e: Fraction(1, len(labs[s])))
            st.markdown("**Árbol de posibilidades** (cada hoja es un resultado distinto)")
            dibujar_arbol(arbol, "Fracción", key="cnt_arbol", mostrar_acum=False)
        else:
            st.caption("El árbol se dibuja cuando hay hasta 60 resultados y 4 etapas.")
        reporte(f"Por el principio multiplicativo, el número total de resultados posibles es {' × '.join(map(str, etapas))} = {res:,}.",
                "cnt_m", "Conteo")
        return

    if tema.startswith("Permutación con elementos repetidos"):
        palabra = st.text_input("Escribe los elementos (p. ej. una palabra) o frecuencias separadas por coma", "MISSISSIPPI", key="cnt_pr")
        if "," in palabra:
            frec = [int(v) for v in palabra.split(",") if v.strip()]
            elementos = None
        else:
            cont = pd.Series(list(palabra.replace(" ", ""))).value_counts()
            frec = cont.tolist()
            elementos = palabra.replace(" ", "")
            st.dataframe(pd.DataFrame({"Elemento": cont.index, "Repeticiones nᵢ": cont.values}).T)
        n = sum(frec)
        res = math.factorial(n)
        for f in frec:
            res //= math.factorial(f)
        st.latex(rf"\frac{{n!}}{{n_1!\,n_2!\cdots}} = \frac{{{n}!}}{{" + r"\,".join(f"{f}!" for f in frec) + rf"}} = {res:,}")
        if elementos and res <= 2000:
            with st.expander(f"Ver los {res} arreglos distintos"):
                arreglos = sorted(set("".join(p) for p in itertools.permutations(elementos))) if len(elementos) <= 9 else None
                st.write(", ".join(arreglos) if arreglos else "Demasiados para listar.")
        reporte(f"Hay {res:,} arreglos distintos de los {n} elementos considerando las repeticiones.", "cnt_pr", "Conteo")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    n = int(c1.number_input("n (elementos disponibles)", 1, 170, 5, key="cnt_n"))
    usa_r = not (tema.startswith("Permutación (todos") or tema.startswith("Permutación circular"))
    sin_rep = "sin repetición" in tema
    r = int(c2.number_input("r (elementos elegidos)", 0, n if sin_rep else 170, min(3, n), key="cnt_r")) if usa_r else n
    nombres = c3.text_input("Nombres de los elementos (opcional, para listar)", ",".join("ABCDEFGHIJ"[:min(n, 10)]), key="cnt_nom")

    if tema.startswith("Permutación (todos"):
        res, formula, gen = math.factorial(n), rf"n! = {n}! = {math.factorial(n):,}", lambda el: itertools.permutations(el)
    elif tema.startswith("Permutación / variación"):
        res = math.perm(n, r)
        formula = rf"P(n,r) = \frac{{n!}}{{(n-r)!}} = \frac{{{n}!}}{{{n - r}!}} = {res:,}"
        gen = lambda el: itertools.permutations(el, r)
    elif tema.startswith("Variación con"):
        res, formula, gen = n ** r, rf"n^r = {n}^{{{r}}} = {n ** r:,}", lambda el: itertools.product(el, repeat=r)
    elif tema.startswith("Combinación sin"):
        res = math.comb(n, r)
        formula = rf"C(n,r) = \binom{{n}}{{r}} = \frac{{n!}}{{r!(n-r)!}} = \frac{{{n}!}}{{{r}!\,{n - r}!}} = {res:,}"
        gen = lambda el: itertools.combinations(el, r)
    elif tema.startswith("Combinación con"):
        res = math.comb(n + r - 1, r)
        formula = rf"CR(n,r) = \binom{{n+r-1}}{{r}} = \binom{{{n + r - 1}}}{{{r}}} = {res:,}"
        gen = lambda el: itertools.combinations_with_replacement(el, r)
    else:
        res = math.factorial(n - 1)
        formula = rf"P_c(n) = (n-1)! = {n - 1}! = {res:,}"
        gen = None
    st.latex(formula)
    st.metric("Resultado", f"{res:,}")

    elementos = [e.strip() for e in nombres.split(",") if e.strip()]
    if gen and len(elementos) == n and res <= 5000:
        lista = ["".join(t) if all(len(x) == 1 for x in t) else "(" + ", ".join(t) + ")" for t in gen(elementos)]
        with st.expander(f"📋 Ver el listado de los {len(lista)} resultados"):
            st.write(", ".join(lista))
        if tema.startswith(("Permutación / variación", "Variación con", "Permutación (todos")) and res <= 60 and r <= 4:
            st.markdown("**Árbol de posibilidades** (en cada etapa se elige un elemento)")
            k = r
            rep = tema.startswith("Variación con")
            arbol = construir_arbol([f"Posición {i + 1}" for i in range(k)],
                                    lambda s, c: [e for e in elementos if rep or e not in c],
                                    lambda s, c, e: Fraction(1, n if rep else n - len(c)))
            dibujar_arbol(arbol, "Fracción", key="cnt_arbol2", mostrar_acum=False)
    elif gen and res > 5000:
        st.caption("El listado se muestra cuando hay hasta 5000 resultados y escribes exactamente n nombres.")

    rfun = {"Permutación (todos": f"factorial({n})", "Permutación / variación": f"factorial({n}) / factorial({n - r})",
            "Variación con": f"{n}^{r}", "Combinación sin": f"choose({n}, {r})", "Combinación con": f"choose({n + r - 1}, {r})",
            "Permutación circular": f"factorial({n - 1})"}
    codigo_r(next(v for k_, v in rfun.items() if tema.startswith(k_)) + "\n# Listado: combn(letters[1:" + str(n) + "], " + str(r) + ")  # combinaciones")
    contexto = st.text_input("Contexto del conteo (opcional)", "las opciones posibles", key="cnt_ctx")
    reporte(f"Usando {tema.split(':')[0].lower()} con n = {n}" + (f" y r = {r}" if usa_r else "") +
            f", existen {res:,} formas distintas de organizar o seleccionar {contexto}.", "cnt", "Conteo")
