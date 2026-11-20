"""¿Qué sismos marca la USGS con alerta de tsunami? — Proyecto Integrador, Grupo 9 (UTN FRM, 2026).

Aplicación de Streamlit con narrativa guiada en barra lateral:
  LA HISTORIA: El proyecto → Los datos → Lo que encontramos → El modelo
  TU TURNO:    Explorá los sismos · Probá el modelo

Lee lo que genera build_app.py en app_data/. Para correrla:  streamlit run app.py
"""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn
import streamlit as st
import streamlit.components.v1 as components

import modelo  # define EsOceanico: joblib la necesita para cargar el pipeline

BASE = Path(__file__).parent
APP_DATA = BASE / "app_data"
MUNDO = alt.topo_feature("https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/world-110m.json", "countries")
alt.data_transformers.disable_max_rows()   # los mapas tienen ~13.600 sismos

# --- Colores: significan lo mismo en toda la aplicación -----------------------------------------
AZUL, AZUL_CLARO = "#2a78d6", "#9cc3ef"          # con marca (claro: con marca y magnitud 6 o más, en el mapa)
GRIS_CLARO, GRIS = "#cfcac3", "#7a7974"          # sin marca · sin marca cargada todavía
NARANJA, VIOLETA = "#d95926", "#8a5cc2"          # falso negativo · falsa alarma
MAR, TIERRA = "#1b9e8f", "#b8862b"               # dónde está el epicentro
NEUTRO = "#4a5560"                               # barras que no son una clase (distribuciones, importancias)
ACENTO = "#1b6b7a"                               # acento editorial (títulos de sección y tarjetas KPI)
PROFUNDIDAD = ["#cfe8c4", "#7fbf6a", "#2e7d32"]  # superficial · intermedio · profundo

RESULTADOS = ["acierto: con marca", "falso negativo", "falsa alarma", "acierto: sin marca", "sin marca cargada"]
ESCALA_RESULTADO = alt.Scale(domain=RESULTADOS, range=[AZUL, NARANJA, VIOLETA, GRIS_CLARO, GRIS])
COLORES_RESULTADO = dict(zip(RESULTADOS, [AZUL, NARANJA, VIOLETA, GRIS_CLARO, GRIS]))
CON, SIN = "con marca de la USGS", "sin marca de la USGS"   # las dos clases, con el recordatorio de qué es la marca
ESCALA_MARCA = alt.Scale(domain=[CON, SIN], range=[AZUL, GRIS_CLARO])
ESCALA_ZONA = alt.Scale(domain=["mar", "tierra"], range=[MAR, TIERRA])

# Regiones donde la USGS usa un umbral regional (sección 2.1 de la Entrega 3)
REGION_ALASKA_OESTE = ["Alaska", "Oregon", "off the coast of Oregon", "Canada", "CA", "California", "OR",
                       "south of Alaska"]

# Rangos de magnitud y profundidad que usan los gráficos de «Lo que encontramos»
CORTES_MAG = [4.99, 5.5, 6.0, 6.5, 7.0, 7.5, 9.5]
RANGOS_MAG = ["5,0–5,5", "5,5–6,0", "6,0–6,5", "6,5–7,0", "7,0–7,5", "más de 7,5"]
CORTES_PROF = [-1, 70, 300, 1000]
RANGOS_PROF = ["superficial (menos de 70 km)", "intermedio (70 a 300 km)", "profundo (más de 300 km)"]

# Enfoques de zoom para el mapa interactivo de «Explorá los sismos»
VISTAS_MAPA = {
    "Todo el mundo": None,
    "Alaska y costa oeste (umbral regional USGS)": {"lat": [25, 72], "lon": [-179, -110]},
    "Pacífico Oeste (Japón, Indonesia, Kermadec)": {"lat": [-45, 55], "lon": [95, 180]},
    "Sudamérica (Chile, Perú, Andes)": {"lat": [-58, 15], "lon": [-95, -55]},
    "Caribe y Centroamérica": {"lat": [5, 28], "lon": [-105, -58]},
}

# Tipos de magnitud de la USGS: el nombre que muestra el selector de «Probá el modelo» y la explicación de su cuadro
NOMBRES_MAGTYPE = {
    "mww": "Magnitud de momento, fase W (la estándar de la USGS)",
    "mb": "Magnitud de ondas de cuerpo (sismos moderados y lejanos)",
    "mw": "Magnitud de momento, informada por otra red",
    "mwr": "Magnitud de momento regional",
    "ml": "Magnitud local, tipo Richter (redes regionales)",
    "mwb": "Magnitud de momento con ondas de cuerpo",
}
EXPLICACION_MAGTYPE = {  # sigla: (cómo se calcula, cuándo la usa la USGS)
    "mww": ("Calcula el momento sísmico (qué tan grande es la falla que se rompió y cuánto se deslizó) con la fase W, "
            "una onda de período muy largo que registran estaciones de todo el mundo.",
            "Es la que la USGS prefiere para la mayoría de los sismos medianos y grandes, y la primera que publica en "
            "los grandes. Por eso es la de casi todos los sismos con marca."),
    "mb": ("Mide la amplitud de las ondas P (las primeras en llegar) en sismógrafos lejanos al sismo.",
           "Sismos moderados y lejanos que todavía no tienen magnitud de momento. Se satura cerca de 6: en sismos "
           "más grandes deja de crecer, así que no sirve para medirlos."),
    "mw": ("Calcula el momento sísmico, pero el dato lo aporta otra agencia o red sismológica.",
           "Cuando la magnitud preferida del catálogo viene de un contribuyente externo, por ejemplo una red de "
           "California."),
    "mwr": ("Calcula el momento sísmico con ondas registradas por estaciones de la región del sismo.",
            "Sismos moderados en zonas con buena red regional (en los datos: Chile, Alaska y Taiwán, entre otras)."),
    "ml": ("Es la escala original de Richter: mide la amplitud de las ondas en estaciones cercanas.",
           "La informan redes regionales (en los datos: Alaska, Puerto Rico, Hawái y Nevada). También se satura en "
           "sismos grandes."),
    "mwb": ("Calcula el momento sísmico con ondas de cuerpo de período largo.",
            "Una alternativa a mww para sismos moderados a grandes."),
}

# Capítulos de la barra lateral (la historia en orden + la parte interactiva)
PAGINAS_HISTORIA = ["El proyecto", "Los datos", "Lo que encontramos", "El modelo"]
PAGINAS_TURNO = ["Explorá los sismos", "Probá el modelo"]
TODAS_PAGINAS = PAGINAS_HISTORIA + PAGINAS_TURNO


# --- Formato en castellano y componentes visuales -----------------------------------------------
def pct(x, dec=0):
    return f"{x * 100:.{dec}f} %".replace(".", ",")


def num(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def miles(n):
    return f"{n:,}".replace(",", ".")


def grafico(chart):
    """Todos los gráficos de Altair van sin el tema de Streamlit, para respetar la paleta de arriba."""
    st.altair_chart(chart, theme=None)


def inyectar_estilos():
    """Estilos livianos adaptables a modo claro y oscuro usando variables de Streamlit."""
    st.markdown(f"""
    <style>
        .kicker {{
            color: {ACENTO};
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }}
        .titulo-historia {{
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.15;
            margin-top: 0;
            margin-bottom: 0.9rem;
            color: var(--text-color);
        }}
        .bajada {{
            font-size: 1.08rem;
            line-height: 1.55;
            color: var(--text-color);
            opacity: 0.8;
            margin-bottom: 1.3rem;
            max-width: 52rem;
        }}
        .cita-veredicto {{
            border-left: 4px solid {ACENTO};
            padding: 0.45rem 0 0.45rem 1.1rem;
            margin: 1.2rem 0 1.6rem 0;
            font-size: 1.18rem;
            line-height: 1.45;
            color: var(--text-color);
            max-width: 52rem;
        }}
        .kpi-card {{
            background-color: var(--secondary-background-color);
            border-radius: 8px;
            padding: 1.15rem 1.25rem;
            height: 100%;
        }}
        .kpi-valor {{
            color: {ACENTO};
            font-size: 2.25rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }}
        .kpi-texto {{
            color: var(--text-color);
            opacity: 0.8;
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        .nav-grupo-historia {{
            color: {ACENTO};
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            border-left: 3px solid {ACENTO};
            padding-left: 0.55rem;
            margin: 1rem 0 0.45rem 0;
        }}
        .nav-grupo-turno {{
            color: #a66a1e;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            border-left: 3px solid #a66a1e;
            padding-left: 0.55rem;
            margin: 1.2rem 0 0.45rem 0;
        }}
    </style>
    """, unsafe_allow_html=True)

def encabezado(kicker, titulo, bajada=None):
    st.markdown(f"<div class='kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='titulo-historia'>{titulo}</div>", unsafe_allow_html=True)
    if bajada:
        st.markdown(f"<div class='bajada'>{bajada}</div>", unsafe_allow_html=True)


def tarjeta_kpi(col, valor, descripcion):
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-valor'>{valor}</div>"
        f"<div class='kpi-texto'>{descripcion}</div></div>",
        unsafe_allow_html=True,
    )


def ir_a(pagina):
    st.session_state["pagina"] = pagina
    st.session_state["_ir_arriba"] = True
    st.session_state["_nav_n"] = st.session_state.get("_nav_n", 0) + 1


def boton_siguiente(siguiente, etiqueta):
    st.divider()
    _, der = st.columns([3, 1.3])
    with der:
        if st.button(f"{etiqueta} →", use_container_width=True, type="primary"):
            ir_a(siguiente)
            st.rerun()


# --- Las funciones de la cátedra (Entrega 2) ----------------------------------------------------
CORTES_SEMAFORO = {"separacion": (0.2, 0.8), "correlacion": (0.2, 0.6)}


def separacion(x, y):
    """Distancia entre dos muestras, en desviaciones estándar."""
    x, y = x.dropna(), y.dropna()
    return abs(x.mean() - y.mean()) / np.sqrt((x.std() ** 2 + y.std() ** 2) / 2)


def semaforo(valor, medida="separacion"):
    bajo, alto = CORTES_SEMAFORO[medida]
    v = abs(valor)
    return "roja" if v < bajo else "verde" if v > alto else "amarilla"


def asociacion(x, y):
    """Pearson, Spearman y la brecha entre ambos."""
    p, s = x.corr(y), x.corr(y, method="spearman")
    return p, s, s - p


# --- Carga de datos y modelo --------------------------------------------------------------------
@st.cache_data
def cargar_sismos():
    return pd.read_csv(APP_DATA / "sismos_app.csv")


@st.cache_data
def cargar_resumen():
    with open(APP_DATA / "resumen.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def cargar_modelo():
    return joblib.load(APP_DATA / "modelo.joblib")


st.set_page_config(page_title="Sismos y marca de tsunami · Grupo 9", layout="wide")
inyectar_estilos()

if not (APP_DATA / "sismos_app.csv").exists():
    st.error("Faltan los datos de la aplicación: correr primero `python build_app.py`.")
    st.stop()

sismos = cargar_sismos().round({"latitude": 2, "longitude": 2, "depth": 1})   # menos decimales = página más liviana
resumen = cargar_resumen()
guardado = cargar_modelo()
exp = resumen["experimentos"]
UMBRAL = guardado["umbral"]
modelados = sismos[sismos["conjunto"] != "sin marca cargada"].copy()   # los que usa el modelo (hasta marzo de 2026)
modelados["marca"] = modelados["tsunami"].map({1: CON, 0: SIN})
modelados["rango_mag"] = pd.cut(modelados["mag"], CORTES_MAG, labels=RANGOS_MAG)
con_marca, sin_marca = modelados[modelados["tsunami"] == 1], modelados[modelados["tsunami"] == 0]
chicos = modelados[(modelados["tsunami"] == 1) & (modelados["mag"] < 6)]
recientes = sismos[sismos["conjunto"] == "sin marca cargada"]
grandes_recientes = recientes[recientes["mag"] >= 7.2]
test = resumen["test"]
vc = resumen["validacion_cruzada"]
vc05 = resumen["validacion_cruzada_umbral_05"]
fondo = alt.Chart(MUNDO).mark_geoshape(fill="#eeeeee", stroke="white", strokeWidth=0.5)

if "pagina" not in st.session_state:
    st.session_state["pagina"] = PAGINAS_HISTORIA[0]

if guardado.get("version_sklearn") != sklearn.__version__:
    st.warning(f"El modelo se guardó con scikit-learn {guardado.get('version_sklearn')} y acá corre "
               f"{sklearn.__version__}. Instalar la versión de requirements.txt.")


# --- Barra lateral: LA HISTORIA vs TU TURNO -----------------------------------------------------
with st.sidebar:
    st.caption("Ciencia de Datos · UTN FRM · 2026  \nProyecto Integrador · Grupo 9 · 5K09")

    st.markdown("<div class='nav-grupo-historia'>LA HISTORIA</div>", unsafe_allow_html=True)
    for p in PAGINAS_HISTORIA:
        activa = st.session_state["pagina"] == p
        st.button(p, key=f"nav_{p}", use_container_width=True,
                  type="primary" if activa else "secondary", on_click=ir_a, args=(p,))

    st.markdown("<div class='nav-grupo-turno'>TU TURNO</div>", unsafe_allow_html=True)
    for p in PAGINAS_TURNO:
        activa = st.session_state["pagina"] == p
        st.button(p, key=f"nav_{p}", use_container_width=True,
                  type="primary" if activa else "secondary", on_click=ir_a, args=(p,))

    st.divider()
    st.caption("La historia se lee en orden. En **Tu turno**, las preguntas las hacés vos.")


# =================================================================================================
# CAPÍTULO 1 · EL PROYECTO (PORTADA)
# =================================================================================================
def pagina_proyecto():
    encabezado(
        "EL PROYECTO",
        "¿Qué características de un sismo permiten estimar si está asociado a un tsunami?",
        "Cada sismo del catálogo de la USGS trae su magnitud, profundidad, coordenadas, tipo de medición y si ocurrió "
        "en el mar o en tierra. Si la alerta respondiera sólo a la física del sismo, esos números tendrían que delatarla.",
    )

    en_alaska = int(chicos["region"].isin(REGION_ALASKA_OESTE).sum())
    pct_alaska = en_alaska / len(chicos) if len(chicos) else 0

    st.markdown(
        f"<div class='cita-veredicto'>Con la magnitud y el lugar del sismo se anticipa el "
        f"<b>{pct(test['recall_tsunami'])}</b> de los sismos que la USGS marca con alerta de tsunami —y el lugar pesa "
        f"tanto como la magnitud porque la USGS aplica un criterio regional.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    tarjeta_kpi(
        c1,
        num(test["f1_macro"], 2),
        f"f1 macro en el test, contra {num(resumen['piso_test']['f1_macro'], 2)} del piso que dice siempre «sin marca»",
    )
    tarjeta_kpi(
        c2,
        pct(test["recall_tsunami"]),
        f"de los {test['aciertos_con_marca'] + test['falsos_negativos']} sismos con marca del test detectados "
        f"({test['aciertos_con_marca']} aciertos, umbral {num(UMBRAL, 2)})",
    )
    tarjeta_kpi(
        c3,
        pct(pct_alaska),
        f"de las marcas en sismos menores a 6 ({en_alaska} de {len(chicos)}) están en Alaska y la costa oeste",
    )

    st.markdown("")
    st.caption(
        f"Medido sobre {miles(len(modelados))} sismos de magnitud 5 o más registrados por la USGS entre "
        f"{pd.Timestamp(resumen['datos']['desde']).strftime('%d/%m/%Y')} y el 31/03/2026. "
        f"Los {miles(len(recientes))} sismos de abril a julio de 2026 quedan afuera del entrenamiento porque todavía "
        f"no tienen la marca cargada; la sección siguiente explica por qué."
    )

    st.info(
        "**Aclaración clave sobre `tsunami = 1`:** en el catálogo de la USGS no confirma que haya habido una ola en la "
        "costa, sino que el sismo disparó su **criterio de alerta de tsunami** (que combina magnitud y región). "
        "Todo lo que sigue estudia y predice esa marca."
    )

    boton_siguiente("Los datos", "Seguir con «Los datos»")


# =================================================================================================
# CAPÍTULO 2 · LOS DATOS
# =================================================================================================
def pagina_datos():
    n_total, n_marca = len(sismos), int(sismos["tsunami"].sum())
    encabezado(
        "LOS DATOS",
        f"De {miles(n_total)} sismos fuertes desde 2019, sólo {miles(n_marca)} llevan la marca: "
        f"apenas 1 de cada {round(n_total / n_marca)}",
        "Cada fila es un sismo de magnitud 5 o más bajado de la API de la USGS. Con clases tan desparejas, un modelo "
        f"que diga siempre «sin marca» acierta el {pct(1 - n_marca / n_total)} sin aprender nada: por eso la accuracy "
        "engaña y medimos todo con f1 macro.",
    )

    # Distribución de clases (el primer golpe visual de la sección, igual que en el ejemplo)
    clases = pd.DataFrame({"marca": [SIN, CON], "sismos": [n_total - n_marca, n_marca]})
    clases["texto"] = [f"{miles(int(n))} · {pct(n / n_total, 1)}" for n in clases["sismos"]]
    barras_clases = alt.Chart(clases).mark_bar(size=42, cornerRadiusEnd=4).encode(
        y=alt.Y("marca:N", title=None, sort=[SIN, CON], axis=alt.Axis(labelFontSize=13)),
        x=alt.X("sismos:Q", title="Sismos", scale=alt.Scale(domain=[0, n_total * 1.16])),
        color=alt.Color("marca:N", scale=ESCALA_MARCA, legend=None),
        tooltip=[alt.Tooltip("marca:N", title=" "), alt.Tooltip("sismos:Q")],
    )
    texto_clases = barras_clases.mark_text(align="left", dx=8, fontSize=14, fontWeight="bold", color="#222").encode(
        text="texto:N"
    )
    grafico((barras_clases + texto_clases).properties(width=860, height=145))

    st.markdown("#### Los sismos en el tiempo: réplicas y el corte de marzo de 2026")
    st.markdown(
        "El catálogo trae unos **160 sismos por mes**, pero dos patrones condicionan cómo entrenamos y validamos: "
        "los picos son **secuencias de réplicas** (no son eventos independientes) y **desde abril de 2026 la USGS "
        "todavía no cargó las marcas** (las barras azules caen a cero)."
    )

    mensual = sismos.groupby("mes", as_index=False).agg(sismos=("id", "count"), con_marca=("tsunami", "sum"))
    mensual["fecha"] = pd.to_datetime(mensual["mes"] + "-01")
    picos = mensual.nlargest(2, "sismos").copy()
    picos["etiqueta"] = [f"{r.sismos} sismos: réplicas en {sismos.loc[sismos['mes'] == r.mes, 'region'].mode()[0]}"
                         for r in picos.itertuples()]
    n_recientes = int((sismos["conjunto"] == "sin marca cargada").sum())
    sombra = pd.DataFrame({
        "desde": [pd.Timestamp(modelo.CORTE)],
        "hasta": [mensual["fecha"].max() + pd.offsets.MonthBegin(1)],
        "t": [f"sin marca cargada ({n_recientes} sismos)"],
    })
    eje_t = alt.X("fecha:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    zona_gris = alt.Chart(sombra).mark_rect(color=GRIS, opacity=0.18).encode(x="desde:T", x2="hasta:T")
    linea_t = alt.Chart(mensual).mark_line(color=NEUTRO, point=alt.OverlayMarkDef(color=NEUTRO, size=16)).encode(
        x=eje_t, y=alt.Y("sismos:Q", title="sismos por mes"),
        tooltip=[alt.Tooltip("mes:N"), alt.Tooltip("sismos:Q"), alt.Tooltip("con_marca:Q", title="con marca")],
    )
    etiquetas_t = alt.Chart(picos).mark_text(align="right", dx=-8, dy=-2, fontSize=11, color="#222").encode(
        x="fecha:T", y="sismos:Q", text="etiqueta:N"
    )
    texto_sombra = alt.Chart(sombra).mark_text(align="right", dx=-4, y=12, fontSize=11, color=GRIS).encode(
        x="hasta:T", text="t:N"
    )
    arriba = (zona_gris + linea_t + etiquetas_t + texto_sombra).properties(width=860, height=205)
    barras_t = alt.Chart(mensual).mark_bar(color=AZUL).encode(
        x=eje_t, y=alt.Y("con_marca:Q", title="con marca por mes"),
        tooltip=[alt.Tooltip("mes:N"), alt.Tooltip("con_marca:Q", title="con marca")],
    )
    abajo = (zona_gris + barras_t).properties(width=860, height=105)
    grafico(alt.vconcat(arriba, abajo, spacing=6))

    st.caption(
        f"Por eso el modelo usa únicamente los {miles(len(modelados))} sismos registrados hasta el 31/03/2026 "
        f"y separa entrenamiento y validación por grupos de lugar y mes, evitando mezclar réplicas."
    )

    # Todo el detalle técnico (origen, pipeline, filtros, columnas y fuga) va ordenado dentro de «Cómo lo sabemos»
    with st.expander("Cómo lo sabemos"):
        st.markdown(f"""
**De dónde salen.** De la API pública de la USGS (`earthquake.usgs.gov/fdsnws/event/1/query`), en GeoJSON y sin clave.
Los descarga un pipeline de Airflow con arquitectura Medallón:

1. `wait_for_source`: sensor que verifica si hubo sismos en la ventana; si hay cero, saltea el resto.
2. `extract` (Bronce): guarda el GeoJSON crudo con las fechas de la ventana.
3. `transform` (Plata): aplana el GeoJSON en una tabla de 17 columnas, un sismo por fila.
4. `enrich` (Plata): calcula y agrega nuestra columna propia, `dist_costa_km`.
5. `validate`: frena el pipeline si hay `id` repetidos, menos de 1.000 filas, menos de 5 columnas o columnas vacías.
6. `save` (Oro): hace *upsert* por `id` en `sismos_master.csv` para que las revisiones de la USGS pisen datos viejos.
7. `notify`: avisa por Discord cuando termina.

Esta app usa una copia congelada (`{resumen['datos']['archivo']}`, del {resumen['datos']['desde']} al
{resumen['datos']['hasta']}, {resumen['datos']['columnas']} columnas, {resumen['datos']['ids_duplicados']} duplicados)
para garantizar reproducibilidad.

**Qué es una fila y qué significa `tsunami = 1`.** Cada fila es un sismo de magnitud 5 o más (por debajo de 5 la
probabilidad de tsunami es prácticamente nula). La etiqueta `tsunami = 1` indica que el sismo recibió marca de alerta de
la USGS, **no** que se haya confirmado una ola en la costa.

**Nuestra columna, `dist_costa_km`.** Distancia del epicentro a la costa más cercana (Natural Earth, escala 50 m)
calculada con Haversine y signo por *ray casting* contra los polígonos continentales: negativa en tierra, positiva en el
mar.

**Qué se filtró y qué quedó.**
""")
        st.dataframe(
            pd.DataFrame(resumen["filtros"]).rename(
                columns={"paso": "Paso", "filas": "Sismos", "con_marca": "Con marca"}
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Las columnas del dataset y qué entra al modelo.**")
        filas_columnas = [
            ("mag", "numérica", "Magnitud del sismo", "Entra al modelo"),
            ("latitude, longitude", "numérica", "Ubicación del epicentro", "Entran al modelo"),
            ("depth", "numérica", "Profundidad en km (46 % vale 10 km: valor por defecto de la USGS)",
             "Entra al modelo"),
            ("magType", "categórica", "Método con que se calculó la magnitud (mww, mb, ml…)",
             "Entra al modelo (one-hot)"),
            ("dist_costa_km", "numérica", "Distancia a la costa con signo (columna propia)",
             "Entra como mar o tierra"),
            ("tsunami", "categórica (0/1)", "Marca de alerta de la USGS", "Es lo que se predice"),
            ("sig", "numérica", "«Significancia» del evento según la USGS", "Sale: fuga (contiene la alerta)"),
            ("nst, dmin, gap, rms", "numéricas", "Métricas de la red de estaciones", "Salen: no mejoran el modelo"),
            ("time", "temporal", "Fecha y hora (UTC)", "Sólo arma grupos de validación"),
            ("place", "texto libre", "Descripción del lugar (de ahí sale la región)", "Sale: identificador"),
            ("id, type, status, net", "id / categóricas", "Código, tipo, estado de revisión y red",
             "Salen: id o constantes"),
        ]
        tabla_columnas = pd.DataFrame(filas_columnas, columns=["Columna", "Tipo", "Qué es", "Uso"])
        nulos_por = {c: sismos[c].isna().mean() for c in ["nst", "dmin", "gap", "rms"]}
        tabla_columnas["Nulos"] = np.where(
            tabla_columnas["Columna"] == "nst, dmin, gap, rms",
            f"nst {pct(nulos_por['nst'])}, el resto < 1 %",
            "—",
        )
        st.dataframe(tabla_columnas, hide_index=True, use_container_width=True)

        sep_sig = separacion(con_marca["sig"], sin_marca["sig"])
        _, s_ms, _ = asociacion(modelados["mag"], modelados["sig"])
        con_sig = {a["variante"]: a for a in exp["ablacion"]}["+ sig (fuga)"]
        base_ = {a["variante"]: a for a in exp["ablacion"]}["base"]
        st.markdown(f"""
**La columna descartada por fuga: `sig`.** Separa muy bien las clases (separación {num(sep_sig, 2)}, zona
{semaforo(sep_sig)}), pero porque la USGS la calcula **usando la propia alerta de tsunami**. Además copia a la magnitud
(Spearman {num(s_ms, 2)}). En la ablación da {num(con_sig['f1'])} contra {num(base_['f1'])} sin ella; se descarta por
cómo se construye, no por el puntaje.

**Las demás columnas descartadas:** `status` tiene un único valor, `type` es casi siempre «earthquake» y `net` es «us»
en el {pct((sismos['net'] == 'us').mean(), 1)}; `nst`, `dmin`, `gap` y `rms` tienen nulos de instrumentación y la USGS
las recalcula al revisar el evento.
""")

    boton_siguiente("Lo que encontramos", "Seguir con «Lo que encontramos»")


# =================================================================================================
# CAPÍTULO 3 · LO QUE ENCONTRAMOS
# =================================================================================================
def pagina_hallazgos():
    encabezado(
        "LO QUE ENCONTRAMOS",
        "La magnitud es la única variable física que separa bien, pero ni los sismos más grandes llevan siempre la marca",
        "La intuición física sugiere tres hipótesis: un sismo con alerta de tsunami debería ser más grande, más "
        "superficial y más cercano a la costa. Al contrastarlas con los datos, sólo la magnitud resiste —y aun así "
        "deja excepciones que recién explica el lugar.",
    )

    por_mag = (modelados.groupby(["rango_mag", "en_el_mar"], observed=True)["tsunami"]
               .agg(sismos="size", con_marca="sum").reset_index())
    por_mag["proporcion"] = por_mag["con_marca"] / por_mag["sismos"]
    por_mag["rango_mag"] = por_mag["rango_mag"].astype(str)

    lineas_mag = alt.Chart(por_mag).mark_line(point=alt.OverlayMarkDef(size=75, filled=True), strokeWidth=3).encode(
        x=alt.X("rango_mag:O", sort=RANGOS_MAG, title="Magnitud", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("proporcion:Q", title="Sismos con marca de la USGS", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color("en_el_mar:N", title="Epicentro", scale=ESCALA_ZONA, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("rango_mag:O", title="magnitud"), alt.Tooltip("en_el_mar:N", title="epicentro"),
                 alt.Tooltip("sismos:Q"), alt.Tooltip("con_marca:Q", title="con marca"),
                 alt.Tooltip("proporcion:Q", title="con marca", format=".1%")],
    )
    etiquetas_mag = lineas_mag.mark_text(dy=-13, fontSize=11, fontWeight="bold").encode(
        text=alt.Text("proporcion:Q", format=".0%")
    )
    regla = alt.Chart(pd.DataFrame({"y": [1.0], "texto": ["si la magnitud fuera una regla fija, llegaría acá (100 %)"]}))
    regla_linea = regla.mark_rule(color=GRIS, strokeDash=[6, 4]).encode(y="y:Q")
    regla_texto = regla.mark_text(align="left", x=6, dy=-8, color=GRIS, fontSize=11).encode(y="y:Q", text="texto:N")
    grafico((lineas_mag + etiquetas_mag + regla_linea + regla_texto).properties(width=860, height=320))

    ult = por_mag[por_mag["rango_mag"] == RANGOS_MAG[-1]].set_index("en_el_mar")
    st.markdown(
        f"La proporción con marca pasa de menos de 2 de cada 100 sismos entre 5,0 y 5,5 a "
        f"**{pct(ult.loc['mar', 'proporcion'])}** en sismos de más de 7,5 en el mar (y "
        f"**{pct(ult.loc['tierra', 'proporcion'])}** en tierra, sobre {int(ult.loc['tierra', 'sismos'])} casos). "
        "Nunca llega al 100 %."
    )

    # Cálculo de separaciones para el gráfico y el resumen de hipótesis
    mar_ = modelados[modelados["dist_costa_km"] > 0]
    sep_mag = separacion(con_marca["mag"], sin_marca["mag"])
    sep_prof = separacion(con_marca["depth"], sin_marca["depth"])
    sep_dist = separacion(con_marca["dist_costa_km"], sin_marca["dist_costa_km"])
    sep_dist_mar = separacion(
        mar_.loc[mar_["tsunami"] == 1, "dist_costa_km"],
        mar_.loc[mar_["tsunami"] == 0, "dist_costa_km"],
    )
    grandes = modelados[modelados["mag"] > 7.5]
    excepciones = grandes[grandes["tsunami"] == 0]

    st.markdown("#### Ni la profundidad ni la distancia a la costa separan por sí solas")
    seps = pd.DataFrame([
        ("Magnitud", sep_mag, "se usa"),
        ("sig (índice de la USGS)", separacion(con_marca["sig"], sin_marca["sig"]), "fuga: se descarta"),
        ("Distancia a la costa (sólo en el mar)", sep_dist_mar, "se usa"),
        ("Distancia a la costa (todos)", sep_dist, "se usa"),
        ("Profundidad", sep_prof, "se usa"),
    ], columns=["variable", "separacion", "estado"])
    seps["texto"] = [f"{num(v, 2)}" + ("  · fuga: se descarta" if e != "se usa" else "")
                     for v, e in zip(seps["separacion"], seps["estado"])]
    orden_vars = seps["variable"].tolist()
    x_max = 1.65

    # Etiquetas autoexplicativas (sin jerga interna de la cursada)
    etiquetas_sep = ["no separa (< 0,2)", "separa poco (0,2–0,8)", "separa bien (> 0,8)"]
    franjas = pd.DataFrame({"desde": [0, 0.2, 0.8], "hasta": [0.2, 0.8, x_max], "zona": etiquetas_sep})
    fondo_sem = alt.Chart(franjas).mark_rect(opacity=0.35).encode(
        x=alt.X("desde:Q", title="Separación entre promedios (en desvíos estándar)",
                scale=alt.Scale(domain=[0, x_max])),
        x2="hasta:Q",
        color=alt.Color("zona:N", title="Capacidad de separar las clases", legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=etiquetas_sep, range=["#f4b6b0", "#f7e3a1", "#bfe3c0"])),
    )
    barras_sep = alt.Chart(seps).mark_bar(size=22).encode(
        y=alt.Y("variable:N", sort=orden_vars, title=None, axis=alt.Axis(labelFontSize=12)),
        x="separacion:Q",
        color=alt.Color("estado:N", scale=alt.Scale(domain=["se usa", "fuga: se descarta"], range=[NEUTRO, "#b9b6b0"]),
                        legend=None),
        tooltip=["variable:N", alt.Tooltip("separacion:Q", format=".3f"), "estado:N"],
    )
    texto_sep = barras_sep.mark_text(align="left", dx=6, color="#222", fontSize=12).encode(text="texto:N")
    grafico((fondo_sem + barras_sep + texto_sep).properties(width=860, height=230))

    # Veredicto visible de las 3 hipótesis físicas
    tipos_m = modelados.groupby("magType")["tsunami"].agg(sismos="size", con_marca="sum")
    with st.container(border=True):
        st.markdown("#### Qué pasó con nuestras hipótesis iniciales")
        st.markdown(f"""
- **Magnitud — confirmada como tendencia, no como regla:** es la única variable física que separa bien
  ({num(sep_mag, 2)} desvíos), pero no alcanza como umbral fijo: {len(excepciones)} de los {len(grandes)} sismos de
  más de 7,5 ({pct(len(excepciones) / len(grandes), 1)}) no tienen marca.
- **Profundidad — refutada:** los sismos con marca no son más superficiales que el resto (separación
  {num(sep_prof, 2)}): los sismos profundos son raros tanto con marca como sin marca.
- **Distancia a la costa — señal débil:** no distingue por sí sola ({num(sep_dist, 2)}); recién separa algo
  ({num(sep_dist_mar, 2)}) cuando se miran únicamente los sismos ocurridos en el mar.
""")

    st.markdown(
        f"Además influye el **método de medición (`magType`)**: de los {miles(int(tipos_m.loc['mb', 'sismos']))} "
        f"sismos medidos en `mb` sólo {int(tipos_m.loc['mb', 'con_marca'])} tienen marca, contra "
        f"{pct(tipos_m.loc['mww', 'con_marca'] / tipos_m.loc['mww', 'sismos'], 1)} en `mww`.  \n"
        "**La incógnita que dejó la exploración:** ¿por qué hay sismos grandes sin marca y sismos chicos con marca? "
        "Esa respuesta apareció recién al entrenar el modelo: **el lugar**."
    )

    with st.expander("Cómo lo sabemos"):
        st.markdown("**Cómo medimos la separación** (diferencia entre el promedio con marca y sin marca, en desvíos):")
        st.latex(r"\text{separación} = \frac{|\bar{x}_{\text{con marca}} - \bar{x}_{\text{sin marca}}|}"
                 r"{\sqrt{(s^2_{\text{con marca}} + s^2_{\text{sin marca}})/2}}")
        prof = modelados.assign(rango=pd.cut(modelados["depth"], CORTES_PROF, labels=RANGOS_PROF))
        reparto = prof.groupby(["marca", "rango"], observed=True).size().reset_index(name="sismos")
        reparto["parte"] = reparto["sismos"] / reparto.groupby("marca")["sismos"].transform("sum")
        reparto["rango"] = reparto["rango"].astype(str)
        reparto["orden"] = reparto["rango"].map({r_: i for i, r_ in enumerate(RANGOS_PROF)})
        sup = reparto[reparto["orden"] == 0].set_index("marca")["parte"]
        st.markdown(f"""
Un valor menor a 0,2 indica que las dos distribuciones casi se superponen; entre 0,2 y 0,8 hay diferencia moderada;
por encima de 0,8 la variable separa bien. Una separación alta no alcanza para incluir una columna (`sig` se descarta
por fuga de datos), y una baja no impide probarla si tiene sentido físico (la profundidad entró igual al modelo).

**Por qué cae la hipótesis de profundidad:** es cierto que el {pct(sup[CON], 1)} de los sismos con marca son
superficiales (< 70 km), pero también lo es el {pct(sup[SIN], 1)} de los sin marca. Las dos barras de abajo son casi
idénticas, e incluso hay {int((con_marca['depth'] > 300).sum())} sismos de más de 300 km con marca.
""")
        barras_p = alt.Chart(reparto).mark_bar().encode(
            y=alt.Y("marca:N", title=None, sort=[CON, SIN]),
            x=alt.X("parte:Q", title=None, axis=alt.Axis(format="%"), stack="normalize"),
            color=alt.Color("rango:N", title="Profundidad", sort=RANGOS_PROF,
                            scale=alt.Scale(domain=RANGOS_PROF, range=PROFUNDIDAD), legend=alt.Legend(orient="top")),
            order=alt.Order("orden:Q"),
            tooltip=["marca:N", "rango:N", "sismos:Q", alt.Tooltip("parte:Q", format=".1%")],
        )
        grafico(barras_p.properties(width=700, height=110))
        st.markdown(f"""
**Sobre el recorte temporal:** en el análisis exploratorio inicial las hipótesis se midieron sobre los
{miles(len(sismos))} sismos totales; acá se calculan sobre los {miles(len(modelados))} con marca cargada hasta marzo
de 2026. Las separaciones cambian recién en el tercer decimal; la proporción de sismos > 7,5 sin marca pasa de 21,7 %
a {pct(len(excepciones) / len(grandes), 1)}.

**Redundancias:** ningún par de columnas candidatas supera 0,6 de correlación, así que no se descartó ninguna por
redundancia. Los pares más altos son `mag`–`sig` (confirma que `sig` copia a la magnitud) y `dist_costa_km`–`dmin`
(las estaciones están en tierra).
""")

    boton_siguiente("El modelo", "Seguir con «El modelo»")

# =================================================================================================
# CAPÍTULO 4 · EL MODELO
# =================================================================================================
def pagina_modelo():
    f1_elegido = {c["modelo"]: c for c in exp["comparacion"]}["Gradient boosting ajustado (elegido)"]["f1"]
    comparacion = pd.DataFrame(exp["comparacion"])
    piso = comparacion.loc[comparacion["modelo"] == "Piso: siempre «sin marca»", "f1"].iloc[0]

    encabezado(
        "EL MODELO",
        f"Un gradient boosting detecta {test['aciertos_con_marca']} de cada "
        f"{test['aciertos_con_marca'] + test['falsos_negativos']} marcas, y la ubicación pesa tanto como la magnitud",
        "La métrica elegida antes de probar modelos es el f1 macro (le da el mismo peso a la clase chica). Como un "
        "falso negativo cuesta mucho más que una falsa alarma, bajamos el umbral de decisión con validación cruzada.",
    )

    # 1. Comparación de modelos en el flujo principal (como en el ejemplo del profesor)
    st.markdown("#### Los árboles superan ampliamente al piso y a la regresión logística")
    modelos_df = comparacion[~comparacion["modelo"].str.startswith("Piso")].copy()
    modelos_df["bajo"] = modelos_df["f1"] - modelos_df["desvio"]
    modelos_df["alto"] = modelos_df["f1"] + modelos_df["desvio"]
    modelos_df["elegido"] = np.where(modelos_df["modelo"].str.contains("elegido"), "elegido", "probado")
    modelos_df["texto"] = [f"{num(f)} ± {num(d)}" for f, d in zip(modelos_df["f1"], modelos_df["desvio"])]
    orden_modelos = modelos_df.sort_values("f1")["modelo"].tolist()
    color_elegido = alt.Color("elegido:N", scale=alt.Scale(domain=["elegido", "probado"], range=[AZUL, NEUTRO]),
                              legend=None)
    eje_m = alt.Y("modelo:N", sort=orden_modelos, title=None, axis=alt.Axis(labelFontSize=12))
    bigotes = alt.Chart(modelos_df).mark_rule(strokeWidth=2.5).encode(
        x=alt.X("bajo:Q", title="f1 macro (media ± desvío en validación cruzada por grupos)",
                scale=alt.Scale(domain=[0.45, 0.96])),
        x2="alto:Q", color=color_elegido, y=eje_m,
    )
    puntos_m = alt.Chart(modelos_df).mark_circle(size=115).encode(
        x="f1:Q", color=color_elegido, y=eje_m, tooltip=["modelo:N", "texto:N"]
    )
    textos_m = alt.Chart(modelos_df).mark_text(align="left", dx=10, color="#222", fontSize=12).encode(
        x="alto:Q", text="texto:N", y=eje_m
    )
    linea_piso = alt.Chart(pd.DataFrame({"x": [piso], "t": [f"piso: {num(piso)}"]}))
    grafico((
        bigotes + puntos_m + textos_m
        + linea_piso.mark_rule(color=GRIS, strokeDash=[4, 4]).encode(x="x:Q")
        + linea_piso.mark_text(align="left", dx=5, y=10, color=GRIS, fontSize=11).encode(x="x:Q", text="t:N")
    ).properties(width=860, height=210))

    rf = comparacion.set_index("modelo").loc["Random forest"]
    gb = comparacion.set_index("modelo").loc["Gradient boosting"]
    st.markdown(
        f"Random forest y gradient boosting **empatan** dentro del desvío ({num(rf['f1'] - gb['f1'])} contra "
        f"{num(gb['desvio'])}); elegimos gradient boosting porque entrena 3× más rápido y es más liviano. "
        "La regresión logística queda atrás porque sólo puede sumar pesos lineales y no puede recortar una región "
        "geográfica como Alaska."
    )

    

   # 2. Resultado final en el conjunto de test (matriz de confusión 2×2 normalizada por fila)
    st.markdown("#### Qué hizo en el test (visto una sola vez)")
    tp = int(test["aciertos_con_marca"])
    fn = int(test["falsos_negativos"])
    fp = int(test["falsas_alarmas"])
    tn = int(test["verdaderos_negativos"])

    matriz = pd.DataFrame([
        {"real": "Con marca USGS", "pred": "Alerta",     "n": tp, "parte": tp / (tp + fn), "tipo": "Acierto (verdadero positivo)"},
        {"real": "Con marca USGS", "pred": "Sin alerta", "n": fn, "parte": fn / (tp + fn), "tipo": "Falso negativo (error caro)"},
        {"real": "Sin marca USGS", "pred": "Alerta",     "n": fp, "parte": fp / (fp + tn), "tipo": "Falsa alarma (falso positivo)"},
        {"real": "Sin marca USGS", "pred": "Sin alerta", "n": tn, "parte": tn / (fp + tn), "tipo": "Acierto (verdadero negativo)"},
    ])
    matriz["n_txt"] = matriz["n"].map(miles)
    matriz["pct_txt"] = matriz["parte"].map(lambda v: pct(v, 1))

    base_cm = alt.Chart(matriz).encode(
        x=alt.X("pred:N", title="Qué dijo el modelo", sort=["Alerta", "Sin alerta"],
                axis=alt.Axis(orient="top", labelAngle=0, labelFontSize=12, titleFontSize=12)),
        y=alt.Y("real:N", title="Realidad (USGS)", sort=["Con marca USGS", "Sin marca USGS"],
                axis=alt.Axis(labelFontSize=12, titleFontSize=12)),
    )
    celdas = base_cm.mark_rect(stroke="white", strokeWidth=3, cornerRadius=6).encode(
        color=alt.Color("parte:Q", title="% de la fila",
                        scale=alt.Scale(domain=[0, 1], range=["#eef3f6", ACENTO]),
                        legend=None),
        tooltip=[
            alt.Tooltip("real:N", title="Realidad"),
            alt.Tooltip("pred:N", title="Predicción"),
            alt.Tooltip("tipo:N", title="Tipo"),
            alt.Tooltip("n:Q", title="Sismos"),
            alt.Tooltip("parte:Q", title="% de la clase real", format=".1%"),
        ],
    )
    color_txt = alt.condition(alt.datum.parte > 0.55, alt.value("white"), alt.value("#1c1e21"))
    txt_n = base_cm.mark_text(dy=-9, fontSize=18, fontWeight="bold").encode(text="n_txt:N", color=color_txt)
    txt_pct = base_cm.mark_text(dy=12, fontSize=12).encode(text="pct_txt:N", color=color_txt)

    col_cm, col_lectura = st.columns([1.15, 1.35])
    with col_cm:
        grafico((celdas + txt_n + txt_pct).properties(width=390, height=230))
    with col_lectura:
        st.markdown(f"""
Con el umbral ajustado en **{num(UMBRAL, 2)}**:

- **Aciertos con marca ({miles(tp)}):** detecta el **{pct(test['recall_tsunami'], 1)}** de los {miles(tp + fn)} sismos
  marcados del test.
- **Falsos negativos ({miles(fn)}):** es el error caro —sismos que la USGS marcó y el modelo dejó pasar. El detalle de
  cada uno está en *«Cómo lo sabemos»*.
- **Falsas alarmas ({miles(fp)}):** 1 cada {round((tn + fp) / fp)} sismos sin marca. De cada 100 alertas que emite el
  modelo, **{round(test['precision_tsunami'] * 100)}** corresponden a una marca real.
- **Aciertos sin marca ({miles(tn)}):** descarta correctamente el **{pct(tn / (tn + fp), 1)}** de los sismos sin alerta.
""")

    # 3. El gran hallazgo del modelo: importancia + mapa con la «caja de Alaska»
    st.markdown("#### Por qué el lugar pesa tanto como la magnitud: la «caja» de Alaska")
    nombres_col = {"longitude": "longitud", "mag": "magnitud", "latitude": "latitud", "magType": "tipo de magnitud",
                   "dist_costa_km": "mar o tierra", "depth": "profundidad"}
    imp = pd.DataFrame(exp["importancia"])
    imp["columna"] = imp["columna"].map(nombres_col)
    imp["bajo"], imp["alto"] = imp["caida"] - imp["desvio"], imp["caida"] + imp["desvio"]
    orden_imp = imp.sort_values("caida", ascending=False)["columna"].tolist()
    barras_i = alt.Chart(imp).mark_bar(color=NEUTRO).encode(
        y=alt.Y("columna:N", sort=orden_imp, title=None),
        x=alt.X("caida:Q", title="caída de f1 macro al desordenarla"),
        tooltip=["columna:N", alt.Tooltip("caida:Q", format=".3f"), alt.Tooltip("desvio:Q", format=".3f")],
    )
    bigotes_i = alt.Chart(imp).mark_rule(color="#222").encode(
        y=alt.Y("columna:N", sort=orden_imp), x="bajo:Q", x2="alto:Q"
    )

    cortes = exp["arbol"]["cortes"]
    lat_c, lon_c, mag_c = cortes.get("latitude", 40.2), cortes.get("longitude", -116.6), cortes.get("mag", 6.05)
    mapa = modelados.assign(
        grupo=lambda d: d["tsunami"].map({0: "sin marca", 1: "con marca, magnitud 6 o más"}).where(
            ~((d["tsunami"] == 1) & (d["mag"] < 6)), "con marca, magnitud menor a 6"
        )
    )
    dominio_g = ["sin marca", "con marca, magnitud 6 o más", "con marca, magnitud menor a 6"]
    puntos_g = alt.Chart(mapa[["longitude", "latitude", "grupo", "tsunami", "place", "mag"]]).mark_circle().encode(
        longitude="longitude:Q", latitude="latitude:Q",
        color=alt.Color("grupo:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=dominio_g, range=[GRIS_CLARO, AZUL_CLARO, AZUL])),
        size=alt.Size("grupo:N", legend=None, scale=alt.Scale(domain=dominio_g, range=[3, 14, 26])),
        order=alt.Order("tsunami:Q"),
        tooltip=[alt.Tooltip("place:N", title="lugar"), alt.Tooltip("mag:Q", title="magnitud")],
    )
    lados = pd.DataFrame({"lon": [-179.9, lon_c, -179.9, -179.9], "lat": [lat_c, lat_c, 75, lat_c],
                          "lon2": [lon_c, lon_c, lon_c, -179.9], "lat2": [lat_c, 75, 75, 75]})
    caja = alt.Chart(lados).mark_rule(color="#e3242b", strokeWidth=2, strokeDash=[6, 3]).encode(
        longitude="lon:Q", latitude="lat:Q", longitude2="lon2:Q", latitude2="lat2:Q"
    )
    texto_caja = alt.Chart(pd.DataFrame({"lon": [lon_c + 3], "lat": [72.0], "t": ["la «caja» del árbol"]})).mark_text(
        align="left", color="#e3242b", fontSize=12, fontWeight="bold"
    ).encode(longitude="lon:Q", latitude="lat:Q", text="t:N")

    izq, der = st.columns([1, 1.85])
    with izq:
        grafico((barras_i + bigotes_i).properties(width=320, height=235))
    with der:
        grafico(alt.layer(fondo, puntos_g, caja, texto_caja).project(type="equirectangular")
                .properties(width=600, height=305))

    st.markdown(
        f"Los puntos azul oscuro son los **{len(chicos)} sismos con marca y magnitud menor a 6**: casi todos ocurren "
        "en Alaska y la costa oeste norteamericana, donde la USGS opera un sistema de alerta con umbral más bajo. "
        f"Incluso un árbol de sólo tres preguntas (f1 {num(exp['arbol']['f1'])}) dibuja esa caja roja: si la magnitud "
        f"supera {num(mag_c, 2)} suele haber marca; si no, la hay al norte de {num(lat_c, 1)}° y al oeste de "
        f"{num(lon_c, 1)}°. El boosting llega a {num(f1_elegido)} combinando cortes más finos."
    )

    with st.container(border=True):
        st.markdown("#### Qué nos deja la historia")
        st.markdown(f"""
- **Respuesta a la pregunta:** lo que permite anticipar si la USGS marca un sismo con alerta de tsunami es **su
  magnitud y su ubicación** casi por partes iguales. El tipo de magnitud (`magType`) aporta algo; la profundidad, la
  distancia a la costa y las métricas de la red de estaciones no suman sobre eso.
- **Por qué no bastaba la física sola:** el modelo explicó lo que la exploración dejó abierto. La etiqueta no depende
  sólo del tamaño del sismo, sino también del **criterio regional operativo de la USGS** (umbral más bajo en Alaska y
  la costa oeste norteamericana, casi sin marcas en regiones como el Caribe).
- **Para qué sirve y para qué no:** sirve para estimar en segundos si la USGS activaría su marca de alerta usando los
  primeros parámetros del sismo, pero **no predice si llegará una ola real a la costa** ni reemplaza un modelo
  oceanográfico de propagación.
- **Prueba futura a ciegas:** sobre los {miles(len(recientes))} sismos de abril a julio de 2026 que la USGS todavía no
  marcó, el modelo emite alerta en {int(recientes['alerta'].sum())} (incluidos los {len(grandes_recientes)} de magnitud
  ≥ 7,2). Cuando la USGS cargue esas marcas se podrá evaluar el modelo con datos completamente nuevos.
""")

    with st.expander("Cómo lo sabemos"):
        g = resumen["grupos"]
        ablacion = {a["variante"]: a for a in exp["ablacion"]}
        base_f1, base_desvio = ablacion["base"]["f1"], ablacion["base"]["desvio"]
        logi = comparacion.set_index("modelo").loc["Regresión logística"]
        st.markdown(f"""
**La validación por grupos.** Para evitar que réplicas de una misma secuencia queden a la vez en entrenamiento y
validación, armamos `StratifiedGroupKFold` (5 pedazos) agrupando por **celda de 2° × 2° y mes** ({miles(g['cantidad'])}
grupos; el mayor son los {g['mayor_sismos']} sismos de Kermadec en marzo de 2021). El test (20 %) se separó con igual
criterio y se miró una sola vez:
""")
        part = pd.DataFrame(exp["particiones"])
        part["valor"] = [f"{num(f)} ± {num(d)}" for f, d in zip(part["f1"], part["desvio"])]
        st.dataframe(part.pivot(index="modelo", columns="forma", values="valor")[["Al azar", "Por grupo", "Por fecha"]])
        st.markdown(f"""
Las tres particiones dan igual dentro del desvío porque casi todas las réplicas son chicas y sin marca; conservamos la
validación por grupo porque replica el uso real. Sin latitud ni longitud la logística baja de {num(logi['f1'])} a
{num(exp['logistica_sin_ubicacion']['f1'])} y el boosting de {num(base_f1)} a
{num(ablacion['− latitud y longitud']['f1'])}.

**El ajuste (`GridSearchCV`, 36 combinaciones):** ganó tasa de aprendizaje
{num(resumen['hiperparametros']['learning_rate'], 2)}, {resumen['hiperparametros']['max_iter']} árboles,
{resumen['hiperparametros']['max_leaf_nodes']} hojas y sin balancear ({num(f1_elegido)}).

**Elección del umbral en validación cruzada.** Como el f1 es casi plano entre 0,2 y 0,5, bajamos el umbral de 0,5 a
{num(UMBRAL, 2)} para subir el recall de {pct(vc05['recall_tsunami'], 1)} a {pct(vc['recall_tsunami'], 1)} (la
precisión pasa de {pct(vc05['precision_tsunami'], 1)} a {pct(vc['precision_tsunami'], 1)}):
""")
        curva = pd.DataFrame(exp["curva_umbral"]).melt(id_vars="umbral", var_name="medida", value_name="valor")
        medidas = ["recall", "precisión", "f1 macro"]
        lineas_u = alt.Chart(curva).mark_line(strokeWidth=2.5).encode(
            x=alt.X("umbral:Q", title="umbral"),
            y=alt.Y("valor:Q", title="validación cruzada", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("medida:N", title=None, sort=medidas,
                            scale=alt.Scale(domain=medidas, range=["#222222", "#9a9a9a", "#2e7d32"]),
                            legend=alt.Legend(orient="top")),
            tooltip=["umbral:Q", "medida:N", alt.Tooltip("valor:Q", format=".3f")],
        )
        marcas_u = alt.Chart(pd.DataFrame({"umbral": [0.5, UMBRAL], "t": ["antes: 0,5", f"elegido: {num(UMBRAL, 2)}"],
                                           "c": [GRIS, AZUL]}))
        grafico((
            lineas_u
            + marcas_u.mark_rule(strokeDash=[4, 4]).encode(x="umbral:Q", color=alt.Color("c:N", scale=None))
            + marcas_u.mark_text(align="left", dx=4, y=10).encode(
                x="umbral:Q", text="t:N", color=alt.Color("c:N", scale=None)
            )
        ).properties(width=760, height=230))

        st.markdown(f"""
**Resumen en el test** (umbral {num(UMBRAL, 2)}): f1 macro {num(test['f1_macro'])} (VC: {num(vc['f1_macro'])}),
recall {pct(test['recall_tsunami'])}, precisión {pct(test['precision_tsunami'])}, PR-AUC {num(test['pr_auc'], 2)}
contra {num(resumen['piso_test']['pr_auc'], 2)} del piso.

**Ablación de columnas** (la franja gris es ± 1 desvío del modelo base; sólo sacar latitud/longitud sale del empate):
""")
        abl = pd.DataFrame([a for a in exp["ablacion"] if a["variante"] != "base"])
        abl["diferencia"] = abl["f1"] - base_f1
        abl["lectura"] = np.where(abl["diferencia"].abs() > base_desvio, "supera el desvío", "empate")
        franja = alt.Chart(pd.DataFrame({"a": [-base_desvio], "b": [base_desvio]})).mark_rect(
            color="#e8e6e1").encode(x="a:Q", x2="b:Q")
        barras_a = alt.Chart(abl).mark_bar().encode(
            y=alt.Y("variante:N", sort=abl.sort_values("diferencia")["variante"].tolist(), title=None),
            x=alt.X("diferencia:Q", title="diferencia de f1 macro respecto del modelo base"),
            color=alt.Color("lectura:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["supera el desvío", "empate"], range=[NEUTRO, "#b9b6b0"])),
            tooltip=["variante:N", alt.Tooltip("f1:Q", format=".3f"), alt.Tooltip("diferencia:Q", format="+.3f")],
        )
        grafico((franja + barras_a).properties(width=700, height=290))

        casos = sismos[sismos["conjunto"] == "test"]
        columnas_e = {"time": "fecha (UTC)", "place": "lugar", "mag": "magnitud", "depth": "profundidad (km)",
                      "en_el_mar": "epicentro", "probabilidad": "probabilidad"}
        fn_ = casos[casos["resultado"] == "falso negativo"].sort_values("probabilidad")
        st.markdown(f"**Dónde se equivoca.** Los {len(fn_)} falsos negativos del test:")
        st.dataframe(fn_[list(columnas_e)].rename(columns=columnas_e), hide_index=True, use_container_width=True)
        st.markdown(f"""
Son sismos de 6,3 a 6,5 fuera de Alaska (en tierra o profundos), sismos chicos frente a California y Alaska marcados
con umbral regional, y uno en el Caribe. **Limitaciones:** los grupos de 2° × 2° y mes son una aproximación; los sismos
desde abril de 2026 aún no tienen marca; la importancia por permutación corta en 0,5 (aunque el orden no cambia con
{num(UMBRAL, 2)}).
""")

    boton_siguiente("Explorá los sismos", "Ahora tu turno: explorá los sismos")


# =================================================================================================
# TU TURNO · EXPLORÁ LOS SISMOS (INTERACTIVO CON PLOTLY + ENFOQUES GUIADOS)
# =================================================================================================
def _geo_plano(centro=None, escala=1.0, detalle=False):
    """Mapa plano con una única copia del mundo (nunca repite continentes)."""
    cfg = dict(
        projection_type="equirectangular",
        resolution=50 if detalle else 110,
        showframe=False,
        showcountries=True,
        countrycolor="#d4d4d4",
        showland=True,
        landcolor="#f7f6f2",
        showocean=True,
        oceancolor="#edf2f6",
        showcoastlines=True,
        coastlinecolor="#c0c0c0",
        showlakes=True,
        lakecolor="#edf2f6",
        bgcolor="rgba(0,0,0,0)",
    )
    if centro is None:
        cfg["lataxis_range"] = [-60, 80]
        cfg["lonaxis_range"] = [-180, 180]
    else:
        cfg["center"] = centro
        cfg["projection_scale"] = escala
        if abs(centro["lon"]) > 135:
            cfg["projection_rotation"] = dict(lon=centro["lon"])
    return cfg


def _vista_mapa_detalle(df):
    """Calcula centro y escala de zoom para encuadrar un grupo de sismos en el mapa plano."""
    lat_min, lat_max = float(df["latitude"].min()), float(df["latitude"].max())
    lon_min, lon_max = float(df["longitude"].min()), float(df["longitude"].max())
    lat_c = (lat_min + lat_max) / 2
    lon_span = lon_max - lon_min
    if lon_span > 180:
        lons_360 = np.where(df["longitude"] < 0, df["longitude"] + 360, df["longitude"])
        lon_c = float((lons_360.min() + lons_360.max()) / 2)
        if lon_c > 180:
            lon_c -= 360
        lon_span = float(lons_360.max() - lons_360.min())
    else:
        lon_c = (lon_min + lon_max) / 2
    span = max((lat_max - lat_min) * 1.8, lon_span, 18.0)
    escala = float(np.clip(260 / span, 2.0, 6.5))
    return {"lat": lat_c, "lon": lon_c}, escala


def _nombre_principal(serie):
    moda = serie.dropna().mode()
    return moda.iloc[0] if len(moda) else "Océano"


# =================================================================================================
# TU TURNO · EXPLORÁ LOS SISMOS (MAPA PLANO ÚNICO EN 2 NIVELES: ZONAS → CLIC PARA ZOOM A SISMOS)
# =================================================================================================
def pagina_explora():
    encabezado(
        "TU TURNO",
        "Explorá los sismos y dónde acierta o falla el modelo",
        "Para no llenar la pantalla de puntos, el mapa arranca agrupado por zonas. **Hacé clic en cualquier círculo** "
        "para acercarte y ver los sismos individuales de esa región.",
    )

    if "explora_zona_id" not in st.session_state:
        st.session_state["explora_zona_id"] = None

    # 1. Enfoques rápidos
    opciones_enfoque = [
        "Aciertos con marca y errores",
        "Sólo errores del modelo",
        "Sismos 2026 sin marca cargada",
        "Todos los sismos (catálogo completo)",
    ]
    if hasattr(st, "segmented_control"):
        enfoque = st.segmented_control("¿Qué querés inspeccionar?", opciones_enfoque, default=opciones_enfoque[0])
        if not enfoque:
            enfoque = opciones_enfoque[0]
    else:
        enfoque = st.radio("¿Qué querés inspeccionar?", opciones_enfoque, horizontal=True)

    # 2. Filtros limpios en una fila
    f1, f2, f3 = st.columns([1.5, 1.3, 1.0])
    mag_min, mag_max = f1.slider("Rango de magnitud", 5.0, 9.0, (5.0, 9.0), step=0.1, format="%.1f")
    region = f2.selectbox(
        "Ir directo a una región",
        ["Todas (elegir en el mapa)"] + sorted(sismos["region"].dropna().unique()),
    )
    epicentro = f3.radio("Epicentro", ["Todos", "mar", "tierra"], horizontal=True)

    # Aplicación de filtros
    vista = sismos[(sismos["mag"] >= mag_min) & (sismos["mag"] <= mag_max)].copy()
    if enfoque == "Aciertos con marca y errores":
        vista = vista[vista["resultado"].isin(["acierto: con marca", "falso negativo", "falsa alarma"])]
    elif enfoque == "Sólo errores del modelo":
        vista = vista[vista["resultado"].isin(["falso negativo", "falsa alarma"])]
    elif enfoque == "Sismos 2026 sin marca cargada":
        vista = vista[vista["conjunto"] == "sin marca cargada"]

    if region != "Todas (elegir en el mapa)":
        vista = vista[vista["region"] == region]
    if epicentro != "Todos":
        vista = vista[vista["en_el_mar"] == epicentro]

    if len(vista) == 0:
        st.info("No hay sismos que cumplan con esta combinación de filtros. Probá ampliar el rango de magnitud o cambiar "
                "el enfoque.")
        return

    vista["marca_real"] = vista["tsunami"].map({1: "Sí", 0: "No"}).where(
        vista["conjunto"] != "sin marca cargada", "Todavía no cargada"
    )
    vista["prob_txt"] = (vista["probabilidad"] * 100).round(0).astype(int).astype(str) + " %"
    vista["tam_mapa"] = np.clip((vista["mag"] - 4.7) ** 2.2, 1.0, 40.0)

    # Agrupación espacial en celdas de 14° × 14° para la vista general sin ruido
    paso = 14
    vista["zona_id"] = (
        (np.floor(vista["latitude"] / paso) * paso).astype(int).astype(str)
        + "_"
        + (np.floor(vista["longitude"] / paso) * paso).astype(int).astype(str)
    )

    zona_activa = st.session_state["explora_zona_id"]
    en_detalle = False
    titulo_zona = None

    if region != "Todas (elegir en el mapa)":
        en_detalle = True
        datos_mapa = vista
        titulo_zona = region
    elif zona_activa is not None:
        sub_zona = vista[vista["zona_id"] == zona_activa]
        if sub_zona.empty:
            st.session_state["explora_zona_id"] = None
            datos_mapa = vista
        else:
            en_detalle = True
            datos_mapa = sub_zona
            titulo_zona = _nombre_principal(sub_zona["region"])
    else:
        datos_mapa = vista

    # 3. Barra de contexto cuando estamos con zoom en una zona
    if en_detalle:
        b1, b2 = st.columns([3, 1.2])
        b1.markdown(f"#### Zona enfocada: **{titulo_zona}** ({miles(len(datos_mapa))} sismos)")
        if region == "Todas (elegir en el mapa)":
            if b2.button("← Volver al mapa por zonas", use_container_width=True, type="primary"):
                st.session_state["explora_zona_id"] = None
                st.rerun()
        else:
            b2.caption("Elegí «Todas» arriba para volver a la vista mundial.")
    else:
        ver_sueltos = st.toggle("Ver todos los puntos sin agrupar", value=False)
        if ver_sueltos:
            en_detalle = True
            datos_mapa = vista

    # 4. KPIs dinámicos sincronizados
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Sismos visibles", miles(len(datos_mapa)))
    k2.metric("Con marca USGS", miles(int((datos_mapa["marca_real"] == "Sí").sum())))
    k3.metric("Alertas del modelo", miles(int(datos_mapa["alerta"].sum())),
              help=f"Probabilidad ≥ {num(UMBRAL, 2)}")
    k4.metric("Magnitud máxima", num(datos_mapa["mag"].max(), 1))

    # 5A. NIVEL 1: MAPA PLANO POR ZONAS (una sola copia del mundo, clic para hacer zoom)
    if not en_detalle:
        zonas = (
            vista.groupby("zona_id", as_index=False)
            .agg(
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
                sismos=("mag", "size"),
                con_marca=("marca_real", lambda s: int((s == "Sí").sum())),
                alertas=("alerta", "sum"),
                errores=("resultado", lambda s: int(s.isin(["falso negativo", "falsa alarma"]).sum())),
                mag_max=("mag", "max"),
                nombre=("region", _nombre_principal),
            )
        )
        zonas["tasa_alerta"] = zonas["alertas"] / zonas["sismos"]
        tam_burbuja = np.interp(np.sqrt(zonas["sismos"]), [1, np.sqrt(max(zonas["sismos"].max(), 2))], [18, 44])

        custom_z = np.stack([
            zonas["zona_id"],
            zonas["nombre"],
            zonas["sismos"].map(miles),
            zonas["con_marca"].map(miles),
            zonas["errores"].map(miles),
            zonas["mag_max"].map(lambda v: num(v, 1)),
        ], axis=-1)

        fig = go.Figure(go.Scattergeo(
            lat=zonas["lat"],
            lon=zonas["lon"],
            mode="markers+text",
            text=zonas["sismos"].astype(str),
            textfont=dict(size=10, color="white"),
            textposition="middle center",
            marker=dict(
                size=tam_burbuja,
                color=zonas["tasa_alerta"],
                colorscale=[[0.0, "#7b8fa1"], [0.45, AZUL], [1.0, ACENTO]],
                cmin=0,
                cmax=1,
                opacity=0.9,
                line=dict(width=1, color="white"),
                colorbar=dict(title="% alerta", tickformat=".0%", len=0.65, thickness=12),
            ),
            customdata=custom_z,
            hovertemplate=(
                "<b>Zona: %{customdata[1]}</b><br>"
                "Sismos en esta zona: <b>%{customdata[2]}</b><br>"
                "Con marca USGS: %{customdata[3]} · Errores del modelo: %{customdata[4]}<br>"
                "Magnitud máxima: %{customdata[5]}<br>"
                "<i>Hacé clic para hacer zoom y ver cada sismo</i><extra></extra>"
            ),
            name="Zonas",
        ))
        fig.update_layout(
            geo=_geo_plano(),
            height=470,
            margin=dict(l=0, r=0, t=5, b=0),
            clickmode="event+select",
        )

        evento = st.plotly_chart(
            fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="mapa_explora_zonas",
        )
        puntos = evento.get("selection", {}).get("points", []) if isinstance(evento, dict) else []
        if puntos:
            pt = puntos[0]
            zona_click = None
            cd = pt.get("customdata")
            if isinstance(cd, (list, tuple)) and len(cd) > 0:
                zona_click = cd[0]
            elif pt.get("point_index") is not None and 0 <= pt["point_index"] < len(zonas):
                zona_click = zonas.iloc[pt["point_index"]]["zona_id"]
            if zona_click:
                st.session_state["explora_zona_id"] = zona_click
                st.rerun()

        st.caption(
            f"Se muestran **{len(zonas)} zonas** que resumen {miles(len(vista))} sismos. "
            "Hacé clic en cualquier círculo para acercarte y explorar sismo por sismo."
        )

    # 5B. NIVEL 2: MAPA PLANO CON ZOOM Y SISMOS INDIVIDUALES
    else:
        if titulo_zona is None:
            geo_cfg = _geo_plano()
        else:
            centro, escala = _vista_mapa_detalle(datos_mapa)
            geo_cfg = _geo_plano(centro=centro, escala=escala, detalle=True)

        orden_capas = ["acierto: sin marca", "sin marca cargada", "falsa alarma", "acierto: con marca", "falso negativo"]
        fig = go.Figure()
        for res in orden_capas:
            sub = datos_mapa[datos_mapa["resultado"] == res]
            if sub.empty:
                continue
            opacidad = 0.35 if res == "acierto: sin marca" else 0.9
            borde = 0.0 if res == "acierto: sin marca" else 0.7
            custom = np.stack([
                sub["place"].fillna("—"),
                sub["time"].str.slice(0, 16),
                sub["mag"].map(lambda v: num(v, 1)),
                sub["magType"].fillna("—"),
                sub["depth"].map(lambda v: num(v, 0)),
                sub["en_el_mar"],
                sub["marca_real"],
                sub["prob_txt"],
                sub["resultado"],
                sub["conjunto"],
            ], axis=-1)
            fig.add_trace(go.Scattergeo(
                lat=sub["latitude"],
                lon=sub["longitude"],
                name=f"{res} ({miles(len(sub))})",
                mode="markers",
                marker=dict(
                    size=np.interp(sub["tam_mapa"], [1, 40], [7, 20]),
                    color=COLORES_RESULTADO[res],
                    opacity=opacidad,
                    line=dict(width=borde, color="white"),
                ),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Fecha: %{customdata[1]} UTC<br>"
                    "Magnitud: <b>%{customdata[2]}</b> (%{customdata[3]}) · Profundidad: %{customdata[4]} km<br>"
                    "Epicentro: %{customdata[5]}<br>"
                    "Marca USGS: <b>%{customdata[6]}</b><br>"
                    "Probabilidad modelo: <b>%{customdata[7]}</b> (%{customdata[8]})<br>"
                    "Conjunto: %{customdata[9]}<extra></extra>"
                ),
            ))

        fig.update_layout(
            geo=geo_cfg,
            height=470,
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        )
        st.plotly_chart(fig, use_container_width=True, key="mapa_explora_detalle")
        st.caption(
            "Podés usar la rueda del mouse para acercarte más, arrastrar el mapa o hacer clic en la leyenda "
            "para prender y apagar resultados."
        )

    # 6. Distribución por magnitud sincronizada
    col_hist, col_info = st.columns([2.2, 1])
    with col_hist:
        fig_hist = px.histogram(
            datos_mapa,
            x="mag",
            color="resultado",
            nbins=25,
            color_discrete_map=COLORES_RESULTADO,
            category_orders={"resultado": RESULTADOS},
            labels={"mag": "Magnitud", "count": "Sismos", "resultado": "Resultado"},
            title="Distribución por magnitud" + (f" · {titulo_zona}" if titulo_zona else " (todas las zonas)"),
        )
        fig_hist.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=35, b=0),
            barmode="stack",
            showlegend=False,
            yaxis_title="Sismos",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_info:
        st.markdown("##### Qué mirar acá")
        if enfoque == "Sismos 2026 sin marca cargada":
            st.markdown(
                f"Son los **{miles(len(recientes))} sismos** desde abril de 2026 sin marca de la USGS. El modelo "
                f"lanza alerta en **{int(recientes['alerta'].sum())}**, incluidos los "
                f"**{len(grandes_recientes)}** de magnitud ≥ 7,2 "
                f"({', '.join(f'{r.region} {num(r.mag, 1)}' for r in grandes_recientes.itertuples())})."
            )
        else:
            st.markdown(
                "- **Clic en un círculo** del mapa general para hacer zoom directo a esa zona.\n"
                "- **Probá cliquear Alaska** para ver la concentración de sismos chicos marcados.\n"
                "- **Cambiá a «Sólo errores»** arriba para detectar dónde le cuesta al modelo."
            )

    # 7. Tabla explorable sincronizada
    with st.expander(f"Ver detalle tabular ({miles(len(datos_mapa))} sismos, ordenados por magnitud)"):
        cols = ["time", "place", "mag", "magType", "depth", "en_el_mar", "marca_real", "prob_txt", "resultado",
                "conjunto"]
        renombre = {
            "time": "Fecha (UTC)", "place": "Lugar", "mag": "Magnitud", "magType": "Tipo",
            "depth": "Profundidad (km)", "en_el_mar": "Epicentro", "marca_real": "Marca USGS",
            "prob_txt": "Prob. modelo", "resultado": "Resultado", "conjunto": "Conjunto",
        }
        st.dataframe(
            datos_mapa.sort_values(["mag", "probabilidad"], ascending=False)[cols].head(300).rename(columns=renombre),
            hide_index=True,
            use_container_width=True,
        )

    boton_siguiente("Probá el modelo", "Probar el modelo con tus propios datos")


# =================================================================================================
# TU TURNO · PROBÁ EL MODELO (MAPA PLANO ÚNICO CLICABLE CON ZOOM POR ZONA)
# =================================================================================================
def pagina_proba():
    encabezado(
        "TU TURNO",
        "Armá un sismo y probá qué probabilidad le da el modelo",
        "Partí de un sismo real, **hacé clic en una zona del mapa** para acercarte y ubicar tu sismo ahí, o ajustá "
        "los controles a mano. Todos los rangos se mantienen dentro de lo que vio el entrenamiento.",
    )

    r = resumen["rangos"]
    tipos = resumen["magtypes"]

    def dentro(valor, rango_):
        return float(min(max(float(valor), float(rango_[0])), float(rango_[1])))

    defaults = {
        "proba_mag": dentro(7.0, r["mag"]),
        "proba_depth": dentro(25.0, r["depth"]),
        "proba_lat": dentro(-33.5, r["latitude"]),
        "proba_lon": dentro(-72.0, r["longitude"]),
        "proba_mar": "mar",
        "proba_magtype": "mww" if "mww" in tipos else tipos[0],
        "proba_zoom_zona": None,
        "proba_ejemplo_prev": "— armarlo desde cero —",
        "proba_mapa_ver": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    pend = st.session_state.pop("_proba_pendiente", None)
    if pend:
        for k, v in pend.items():
            st.session_state[k] = v

    def ejemplo(filtro, por, descripcion):
        """Devuelve el sismo real con el mayor valor de `por` entre los que cumplen el filtro."""
        candidatos = modelados[filtro(modelados)]
        return (descripcion, candidatos.loc[candidatos[por].idxmax()]) if len(candidatos) else None

    ejemplos = [e for e in [
        ejemplo(lambda d: d["mag"] > 0, "mag", "El más grande del catálogo"),
        ejemplo(lambda d: (d["region"] == "Alaska") & (d["tsunami"] == 1) & (d["mag"] < 5.5), "probabilidad",
                "Un sismo chico de Alaska, con marca"),
        ejemplo(lambda d: (d["en_el_mar"] == "tierra") & (d["mag"] >= 6.8), "mag", "Un sismo fuerte en tierra"),
        ejemplo(lambda d: d["depth"] > 300, "mag", "Un sismo grande y profundo"),
        ejemplo(lambda d: (d["magType"] == "mb") & (d["tsunami"] == 0), "mag", "El mayor sismo informado en mb"),
    ] if e]
    opciones = ["— armarlo desde cero —"] + [
        f"{d}: {s['place']} ({num(s['mag'], 1)}, {s['time'][:10]})" for d, s in ejemplos
    ]
    elegido = st.selectbox("Partir de un sismo real", opciones)
    real = ejemplos[opciones.index(elegido) - 1][1] if elegido != opciones[0] else None

    if elegido != st.session_state["proba_ejemplo_prev"]:
        st.session_state["proba_ejemplo_prev"] = elegido
        if real is not None:
            st.session_state["proba_mag"] = dentro(round(float(real["mag"]), 1), r["mag"])
            st.session_state["proba_depth"] = dentro(round(float(real["depth"])), r["depth"])
            st.session_state["proba_lat"] = dentro(round(float(real["latitude"]), 1), r["latitude"])
            st.session_state["proba_lon"] = dentro(round(float(real["longitude"]), 1), r["longitude"])
            st.session_state["proba_mar"] = "mar" if real["en_el_mar"] == "mar" else "tierra"
            st.session_state["proba_magtype"] = real["magType"] if real["magType"] in tipos else defaults["proba_magtype"]
            st.session_state["proba_zoom_zona"] = str(real["region"]) if pd.notna(real["region"]) else "Sismo elegido"
            st.session_state["proba_mapa_ver"] += 1
        else:
            st.session_state["proba_zoom_zona"] = None
            st.session_state["proba_mapa_ver"] += 1

    if real is not None:
        st.caption(
            f"En la realidad, la USGS {'**lo marcó**' if real['tsunami'] == 1 else '**no lo marcó**'}. "
            "Los controles arrancan con sus valores (redondeados): movelos o cliqueá otro punto en el mapa."
        )

    p1, p2 = st.columns(2)
    mag = p1.slider("Magnitud", float(r["mag"][0]), float(r["mag"][1]), step=0.1, format="%.1f", key="proba_mag")
    depth = p1.slider("Profundidad (km)", float(r["depth"][0]), float(r["depth"][1]), step=1.0, key="proba_depth")
    lat = p2.slider("Latitud", float(r["latitude"][0]), float(r["latitude"][1]), step=0.1, format="%.1f",
                    key="proba_lat")
    lon = p2.slider("Longitud", float(r["longitude"][0]), float(r["longitude"][1]), step=0.1, format="%.1f",
                    key="proba_lon")
    en_el_mar = p2.radio("¿El epicentro está en el mar o en tierra?", ["mar", "tierra"], horizontal=True,
                         key="proba_mar") == "mar"

    with st.expander("Opciones avanzadas: tipo de magnitud"):
        mag_type = st.selectbox(
            "Tipo de magnitud", tipos,
            format_func=lambda m: f"{NOMBRES_MAGTYPE.get(m, m)} ({m})",
            help="Sólo se ofrecen los tipos con al menos 20 sismos en el entrenamiento.",
            key="proba_magtype",
        )
        cuota_mww = (modelados["magType"] == "mww").mean()
        st.caption(
            f"Método con que la USGS calculó la magnitud. Si no sabés cuál elegir, dejá *mww*: representa "
            f"{round(cuota_mww * 10)} de cada 10 sismos y es el estándar para sismos grandes."
        )

    if mag_type == "mb" and mag >= 6.5:
        todos_mb = sismos[sismos["magType"] == "mb"]
        st.warning(
            f"Combinación casi inexistente: la escala *mb* se satura cerca de 6, así que la USGS casi no informa en "
            f"mb sismos tan grandes ({(todos_mb['mag'] >= 6.5).sum()} de {miles(len(todos_mb))}; máximo "
            f"{num(todos_mb['mag'].max(), 1)}). Tomá esta probabilidad con cuidado."
        )

    probabilidad = guardado["pipeline"].predict_proba(modelo.caso(mag, depth, lat, lon, mag_type, en_el_mar))[0, 1]
    alerta = probabilidad >= UMBRAL

    m1, m2 = st.columns(2)
    tarjeta_kpi(m1, pct(probabilidad), "probabilidad de que la USGS le ponga la marca de alerta")
    tarjeta_kpi(m2, "Sí · Alerta" if alerta else "No · Sin alerta",
                f"decisión del modelo con el umbral elegido ({num(UMBRAL, 2)})")

    rango_elegido = RANGOS_MAG[int(np.digitize(mag, CORTES_MAG[1:-1], right=True))]
    mismo_rango = modelados[modelados["rango_mag"].astype(str) == rango_elegido]
    misma_zona = mismo_rango[mismo_rango["en_el_mar"] == ("mar" if en_el_mar else "tierra")]
    st.markdown("")
    st.caption(
        f"Referencia histórica: en los datos, los sismos de magnitud {rango_elegido} tienen la marca el "
        f"{pct(mismo_rango['tsunami'].mean())} de las veces ({pct(misma_zona['tsunami'].mean())} en "
        f"{'el mar' if en_el_mar else 'tierra'}). La diferencia con la predicción la explica, sobre todo, el lugar."
    )

    # --- Mapa plano en 2 niveles sin continentes duplicados ---
    zona_enfocada = st.session_state["proba_zoom_zona"]
    b1, b2 = st.columns([3, 1.3])
    if zona_enfocada is None:
        b1.markdown("#### Ubicación en el mapa · *hacé clic en una zona para acercarte*")
        if b2.button("Acercar a mi sismo", use_container_width=True):
            st.session_state["proba_zoom_zona"] = "Ubicación elegida"
            st.session_state["proba_mapa_ver"] += 1
            st.rerun()
    else:
        b1.markdown(f"#### Zoom en **{zona_enfocada}** · *cliqueá un sismo cercano para mover tu punto ahí*")
        if b2.button("← Volver a vista mundial", use_container_width=True, type="primary"):
            st.session_state["proba_zoom_zona"] = None
            st.session_state["proba_mapa_ver"] += 1
            st.rerun()

    cortes = exp["arbol"]["cortes"]
    lat_c, lon_c = cortes.get("latitude", 40.2), cortes.get("longitude", -116.6)
    fig_sim = go.Figure()

    # Caja regional de Alaska como referencia visual
    fig_sim.add_trace(go.Scattergeo(
        lat=[lat_c, lat_c, 74, 74, lat_c],
        lon=[-179.5, lon_c, lon_c, -179.5, -179.5],
        mode="lines",
        name="Zona umbral regional (Alaska / Costa Oeste)",
        line=dict(color="#e3242b", width=1.8, dash="dash"),
        hoverinfo="skip",
    ))

    if zona_enfocada is None:
        # NIVEL 1: Mapa plano mundial único con zonas sísmicas clicables
        paso = 22
        tmp = modelados.assign(
            zid=(np.floor(modelados["latitude"] / paso) * paso).astype(int).astype(str)
            + "_"
            + (np.floor(modelados["longitude"] / paso) * paso).astype(int).astype(str)
        )
        zonas = (
            tmp.groupby("zid", as_index=False)
            .agg(
                lat=("latitude", "mean"),
                lon=("longitude", "mean"),
                sismos=("mag", "size"),
                con_marca=("tsunami", "sum"),
                nombre=("region", _nombre_principal),
                mar=("en_el_mar", lambda s: "mar" if (s == "mar").mean() >= 0.5 else "tierra"),
            )
        )
        zonas = zonas[zonas["sismos"] >= 20].copy()
        zonas["tasa"] = zonas["con_marca"] / zonas["sismos"]
        tam_z = np.interp(np.sqrt(zonas["sismos"]), [1, np.sqrt(max(zonas["sismos"].max(), 2))], [14, 30])
        custom_z = np.stack([
            np.full(len(zonas), "zona"),
            zonas["nombre"],
            zonas["lat"].round(1).astype(str),
            zonas["lon"].round(1).astype(str),
            zonas["mar"],
            zonas["sismos"].map(miles),
            zonas["tasa"].map(lambda v: pct(v, 1)),
        ], axis=-1)

        fig_sim.add_trace(go.Scattergeo(
            lat=zonas["lat"],
            lon=zonas["lon"],
            mode="markers",
            name="Zonas sísmicas (clic para probar ahí)",
            marker=dict(size=tam_z, color="#6d8290", opacity=0.45, line=dict(width=1, color="white")),
            customdata=custom_z,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Sismos registrados: %{customdata[5]} · Con marca: %{customdata[6]}<br>"
                "<i>Hacé clic para mover tu sismo acá y hacer zoom</i><extra></extra>"
            ),
        ))
        geo_cfg = _geo_plano()
    else:
        # NIVEL 2: Zoom en la zona + sismos reales cercanos de magnitud parecida
        dlat = (modelados["latitude"] - lat).abs()
        dlon = ((modelados["longitude"] - lon + 180) % 360 - 180).abs()
        cercanos = modelados[(dlat <= 12) & (dlon <= 16) & ((modelados["mag"] - mag).abs() <= 0.6)].copy()
        if len(cercanos) > 180:
            marcados_c = cercanos[cercanos["tsunami"] == 1]
            no_marcados_c = cercanos[cercanos["tsunami"] == 0].head(max(180 - len(marcados_c), 0))
            cercanos = pd.concat([marcados_c, no_marcados_c])

        for valor_marca, etiqueta_m, color_m, tam_m, opac_m in [
            (0, "Sismos reales cercanos · sin marca", GRIS, 9, 0.55),
            (1, "Sismos reales cercanos · con marca USGS", AZUL_CLARO, 12, 0.9),
        ]:
            sub = cercanos[cercanos["tsunami"] == valor_marca]
            if sub.empty:
                continue
            custom_s = np.stack([
                np.full(len(sub), "sismo"),
                sub["place"].fillna("—"),
                sub["latitude"].round(1).astype(str),
                sub["longitude"].round(1).astype(str),
                sub["en_el_mar"],
                sub["mag"].map(lambda v: num(v, 1)),
                sub["depth"].map(lambda v: num(v, 0)),
                np.full(len(sub), "Sí" if valor_marca == 1 else "No"),
            ], axis=-1)
            fig_sim.add_trace(go.Scattergeo(
                lat=sub["latitude"],
                lon=sub["longitude"],
                mode="markers",
                name=f"{etiqueta_m} ({len(sub)})",
                marker=dict(size=tam_m, color=color_m, opacity=opac_m, line=dict(width=0.7, color="white")),
                customdata=custom_s,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>"
                    "Magnitud real: %{customdata[5]} · Profundidad: %{customdata[6]} km · %{customdata[4]}<br>"
                    "Marca USGS real: <b>%{customdata[7]}</b><br>"
                    "<i>Hacé clic para mover tu sismo a este punto</i><extra></extra>"
                ),
            ))
        geo_cfg = _geo_plano(centro={"lat": lat, "lon": lon}, escala=4.5, detalle=True)

    # Pin del sismo simulado
    fig_sim.add_trace(go.Scattergeo(
        lat=[lat], lon=[lon], mode="markers", name="Tu sismo simulado",
        marker=dict(size=16, color=AZUL if alerta else "#e3242b", line=dict(width=3, color="white")),
        hovertemplate=(
            f"<b>Tu sismo simulado</b><br>"
            f"Lat: {num(lat, 1)} · Lon: {num(lon, 1)} ({'mar' if en_el_mar else 'tierra'})<br>"
            f"Magnitud: {num(mag, 1)} ({mag_type}) · Profundidad: {num(depth, 0)} km<br>"
            f"Probabilidad: <b>{pct(probabilidad)}</b><extra></extra>"
        ),
    ))

    fig_sim.update_layout(
        geo=geo_cfg,
        height=440,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title=None),
        clickmode="event+select",
    )

    clave_mapa = f"mapa_proba_{'mundo' if zona_enfocada is None else 'zoom'}_{st.session_state['proba_mapa_ver']}"
    evento = st.plotly_chart(
        fig_sim,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=clave_mapa,
    )
    puntos = evento.get("selection", {}).get("points", []) if isinstance(evento, dict) else []
    if puntos:
        cd = puntos[0].get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) >= 5 and cd[0] in ("zona", "sismo"):
            nuevo_nombre = cd[1] if cd[0] == "zona" else (zona_enfocada or cd[1])
            st.session_state["_proba_pendiente"] = {
                "proba_lat": dentro(float(cd[2]), r["latitude"]),
                "proba_lon": dentro(float(cd[3]), r["longitude"]),
                "proba_mar": "mar" if cd[4] == "mar" else "tierra",
                "proba_zoom_zona": nuevo_nombre,
                "proba_mapa_ver": st.session_state["proba_mapa_ver"] + 1,
            }
            st.rerun()

    en_mb = modelados[modelados["magType"] == "mb"]
    with st.expander("¿Qué significa cada tipo de magnitud?"):
        conteo_tipos = modelados.groupby("magType")["tsunami"].agg(["size", "sum"])
        filas = []
        for m in tipos:
            como, cuando = EXPLICACION_MAGTYPE.get(m, ("—", "—"))
            n, k = ((int(conteo_tipos.loc[m, "size"]), int(conteo_tipos.loc[m, "sum"]))
                    if m in conteo_tipos.index else (0, 0))
            en_datos = f"{miles(n)} sismos · {pct(k / n, 1)} con marca" if n else "—"
            filas.append(f"| **{m}** | {NOMBRES_MAGTYPE.get(m, m)} | {como} | {cuando} | {en_datos} |")
        st.markdown(f"""
La **magnitud** mide el tamaño de un sismo, pero hay dos familias de cálculo:

- **Magnitudes de momento** (*mw*): miden el momento sísmico (área rota y deslizamiento de la falla) y **no se saturan**.
- **Magnitudes de amplitud** (*mb* y *ml*): miden la amplitud de una onda en el sismógrafo; son rápidas, pero **se
  saturan** en sismos grandes.

| Sigla | Nombre | Cómo se calcula | Cuándo la usa la USGS | En los datos (hasta marzo de 2026) |
|---|---|---|---|---|
""" + "\n".join(filas) + f"""

**Por qué cambia la predicción:** casi ningún sismo informado en *mb* lleva marca ({int(en_mb['tsunami'].sum())} de
{miles(len(en_mb))}), porque la USGS la usa en sismos moderados. Por eso el modelo asocia *mb* con «sin marca».
""")

    st.divider()
    st.caption(
        f"Datos: USGS (earthquake.usgs.gov), sismos del {resumen['datos']['desde']} al {resumen['datos']['hasta']}. "
        f"Modelo: gradient boosting entrenado con datos hasta el 31/03/2026, semilla {resumen['semilla']}."
    )
# =================================================================================================
# ENRUTADOR DE PÁGINAS
# =================================================================================================
RUTAS = {
    "El proyecto": pagina_proyecto,
    "Los datos": pagina_datos,
    "Lo que encontramos": pagina_hallazgos,
    "El modelo": pagina_modelo,
    "Explorá los sismos": pagina_explora,
    "Probá el modelo": pagina_proba,
}
RUTAS[st.session_state["pagina"]]()
if st.session_state.pop("_ir_arriba", False):
    n = st.session_state.get("_nav_n", 0)
    components.html(
        f"""
        <script>
            // nav-{n}
            function subir() {{
                const doc = window.parent.document;
                [
                    doc.querySelector('section[data-testid="stMain"]'),
                    doc.querySelector('[data-testid="stAppViewContainer"]'),
                    doc.querySelector('section.main'),
                    doc.documentElement,
                    doc.body,
                ].forEach(el => {{ if (el) el.scrollTop = 0; }});
                window.parent.scrollTo(0, 0);
            }}
            subir();
            setTimeout(subir, 80);
        </script>
        """,
        height=0,
    )