# 📊 Solver Estadístico Unificado

Herramienta de apoyo para Probabilidad y Estadística (300MAE005): cálculos guiados paso a paso, gráficos interactivos,
conclusiones para informe y el código R equivalente de cada cálculo.

## Módulos

| Sección | Qué incluye |
|---|---|
| Indicadores Estadísticos | Tendencia, dispersión, forma, posición y gráficos |
| **Tabla de Frecuencias** | No agrupada, agrupada (Sturges / √n / manual), cualitativa; fᵢ, Fᵢ, hᵢ, Hᵢ; media, mediana, moda y percentiles agrupados con demostración; histograma, polígono, ojiva, Pareto |
| **Probabilidad Básica** | Unión, intersección, complemento, condicional e independencia con diagrama de Venn sombreable (2 y 3 eventos); tablas de contingencia y χ²; Ley de los Grandes Números |
| Reglas de Conteo | Principio multiplicativo, permutaciones, variaciones, combinaciones, repetidos y circulares, con listado y árbol de posibilidades |
| **Árboles de Probabilidad y Bayes** | Constructor de árboles de 1–4 etapas, ensayos repetidos, urnas con/sin reemplazo y Bayes; eventos por etapa, por conteo, por lista o con clic en el gráfico; demostración de la regla de la multiplicación, probabilidad total y Bayes; árbol invertido; variable aleatoria sobre el árbol; simulación |
| Variable Aleatoria | Discreta (E, Var, F(x), P(a≤X≤b), E(aX+b), E[g(X)], Chebyshev) y continua (hallar k, E, Var, mediana, percentiles por integración) |
| **Variables Aleatorias Conjuntas** | Marginales, condicionales, E(XY), Cov, ρ, independencia, Var(aX+bY), probabilidad de regiones, Z = g(X,Y) |
| Distribuciones Probabilísticas | Binomial, Poisson, Hipergeométrica, Geométrica, Binomial negativa, Uniforme, Normal, Exponencial, Gamma, t, χ², F; gráfico sombreado, demostración, aproximaciones y simulación |
| Distribuciones de muestreo | Media y proporción muestral + simulación del TLC |
| Intervalos de confianza | Media, proporción, varianza (χ²), diferencia de medias y proporciones, tamaño de muestra, simulación de cobertura |
| Pruebas de hipótesis | Media (Z/t), proporción, diferencias, pareadas y varianza (χ²) con gráfico de región de rechazo |
| **Regresión Lineal** | Simple (demostración, inferencia, ANOVA, residuos, bandas, predicción) y múltiple (matricial, R² ajustado, F, VIF, plano 3D) |

## Ejecutar

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

El código nuevo está en la carpeta `pye/` (un archivo por tema del curso).
