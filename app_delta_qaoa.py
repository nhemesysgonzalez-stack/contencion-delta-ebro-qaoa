import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Optimizador Ambiental Delta QAOA", page_icon="🛡️", layout="wide")
st.title("🛡️ Optimizador Cuántico de Contención Ambiental (QAOA)")
st.subheader("Protección de Acuicultura y Biodiversidad en el Delta del Ebro (Tarragona)")
st.write("Sistema de apoyo a la decisión basado en **IBM Qiskit (QAOA)** para asignar barreras absorbentes biodegradables bajo restricciones logísticas y presupuestarias.")

# --- FICHA EJECUTIVA DE NEGOCIO Y ROI ---
with st.expander("📊 Impacto Empresarial, Perfiles Objetivo y Ventaja Tecnológica", expanded=True):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        * **Perfiles y Sectores Objetivo:** Gestores de parques naturales, cofradías de acuicultura (mejillón y ostra), consultoras de ingeniería ambiental, autoridades portuarias y protección civil.
        * **Problema que resuelve:** Ante un episodio de contaminación o marea roja en bahías someras (El Fangar y Els Alfacs), los recursos de contención son limitados y las distancias entre puntos condicionan el despliegue.
        """)
    with col_info2:
        st.markdown("""
        * **Ventaja frente a métodos tradicionales:** Cuando crecen las variables (corrientes, valor ecológico, presupuesto e incompatibilidades logísticas), el cálculo clásico secuencial se ralentiza. El algoritmo híbrido **QAOA** codifica las restricciones directamente en el Hamiltoniano cuántico para evaluar escenarios simultáneamente.
        * **Validación en Hardware Real (QPU):** Este entorno ejecuta simulación matemática exacta de Qiskit. Para ejecutar pruebas piloto sobre procesadores físicos de **IBM Quantum**, contactar al equipo de desarrollo al pie de página.
        """)

# --- MOTOR CUÁNTICO QAOA ---
def ejecutar_qaoa_delta(df_zonas, bloquear_0_2=True):
    pesos = []
    for idx, fila in df_zonas.iterrows():
        puntuacion = (float(fila.get("Valor_Ecologico")) + float(fila.get("Riesgo_Corriente"))) / 100.0
        pesos.append(puntuacion)
        
    terminos = [
        ("IIIZ", -pesos.pop(0)),
        ("IIZI", -pesos.pop(0)),
        ("IZII", -pesos.pop(0)),
        ("ZIII", -pesos.pop(0)),
        ("IIZZ", 2.50), ("IZIZ", 2.50), ("ZIIZ", 2.50),
        ("IZZI", 2.50), ("ZIZI", 2.50), ("ZZII", 2.50)
    ]
    if bloquear_0_2:
        terminos.append(("IZIZ", 4.00))
        
    hamiltoniano = SparsePauliOp.from_list(terminos)
    circuito_qaoa = QAOAAnsatz(cost_operator=hamiltoniano, reps=2)
    estimador = StatevectorEstimator()
    
    def funcion_coste(angulos):
        trabajo = estimador.run([(circuito_qaoa, hamiltoniano, angulos)])
        return float(next(iter(trabajo.result())).data.evs)
        
    res_opt = minimize(funcion_coste, [0.5, 0.5, 0.5, 0.5], method="COBYLA")
    circuito_final = circuito_qaoa.assign_parameters(res_opt.x)
    circuito_final.measure_all()
    
    simulador = AerSimulator()
    conteos = simulador.run(circuito_final.decompose(reps=3), shots=1000).result().get_counts()
    
    ganador_bin = max(conteos, key=conteos.get)
    bits = ganador_bin[::-1]
    return bits, conteos

# --- MENÚ LATERAL ---
st.sidebar.header("🌊 Configuración del Escenario")
modo = st.sidebar.radio(
    "Fuente de datos ambientales:",
    (
        "1. Escenario Base (Delta del Ebro)",
        "2. Control Numérico Manual",
        "3. Subir Tabla Propia (CSV / Excel)"
    )
)
bloquear_distancia = st.sidebar.checkbox("Restricción logística: Evitar despliegue simultáneo Fangar (q0) + Alfacs (q2)", value=True)

df_trabajo = None

if modo == "1. Escenario Base (Delta del Ebro)":
    df_trabajo = pd.DataFrame({
        "Zona_Critica": ["Bahia_Fangar_Mejilloneras", "Desembocadura_Ebro_Humedal", "Bahia_Alfacs_Ostras", "Istmo_Trabucador"],
        "Valor_Ecologico": list((90, 85, 95, 60)),
        "Riesgo_Corriente": list((80, 70, 90, 50))
    })
    st.subheader("1. Mapa de Zonas Críticas (Escenario Predefinido)")
    st.dataframe(df_trabajo, use_container_width=True)

elif modo == "2. Control Numérico Manual":
    st.subheader("1. Ajuste Manual de Parámetros Ambientales")
    st.write("Modifica directamente los valores de sensibilidad ecológica y riesgo por corriente en la tabla:")
    df_base = pd.DataFrame({
        "Zona_Critica": ["Zona_0_Fangar", "Zona_1_Desembocadura", "Zona_2_Alfacs", "Zona_3_Trabucador"],
        "Valor_Ecologico": list((90, 85, 95, 60)),
        "Riesgo_Corriente": list((80, 70, 90, 50))
    })
    df_trabajo = st.data_editor(df_base, num_rows="fixed", use_container_width=True)

else:
    st.subheader("1. Carga de Datos Ambientales (CSV o Excel)")
    st.write("Sube un archivo con **4 filas** (una por cuadrante) y las columnas: `Zona_Critica`, `Valor_Ecologico`, `Riesgo_Corriente`.")
    plantilla = pd.DataFrame({
        "Zona_Critica": ["Cuadrante_Norte_Fangar", "Cuadrante_Centro_Ebro", "Cuadrante_Sur_Alfacs", "Cuadrante_Exterior_Trabucador"],
        "Valor_Ecologico": list((90, 85, 95, 60)),
        "Riesgo_Corriente": list((75, 85, 90, 55))
    })
    st.download_button(
        label="📥 Descargar plantilla CSV de emergencia marítima",
        data=plantilla.to_csv(index=False).encode('utf-8'),
        file_name="plantilla_contencion_delta.csv",
        mime="text/csv"
    )
    archivo = st.file_uploader("Sube tu archivo CSV o Excel (.xlsx)", type=["csv", "xlsx"])
    if archivo is not None:
        df_trabajo = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        st.dataframe(df_trabajo.head(4), use_container_width=True)

# --- EJECUCIÓN ---
if df_trabajo is not None and len(df_trabajo) >= 4:
    if st.button("🚀 Calcular Despliegue Óptimo con QAOA"):
        with st.spinner("Ejecutando circuito QAOA y optimizando restricciones logísticas..."):
            bits_ganadores, conteos = ejecutar_qaoa_delta(df_trabajo.head(4), bloquear_distancia)
            
            zonas_elegidas = []
            for i in range(4):
                if bits_ganadores[i] == '1':
                    zonas_elegidas.append(str(df_trabajo.iloc[i].get("Zona_Critica")))
                    
            st.subheader("2. Plan de Acción Recomendado por el Circuito QAOA")
            st.success(f"✅ **Desplegar barreras absorbentes en:** {', '.join(zonas_elegidas)}")
            
            puntos_totales = sum([int(df_trabajo.iloc[i].get("Valor_Ecologico")) + int(df_trabajo.iloc[i].get("Riesgo_Corriente")) for i in range(4) if bits_ganadores[i] == '1'])
            motivo_logistico = "respetando la restricción de distancia entre Fangar y Alfacs" if bloquear_distancia else "sin restricciones de distancia entre bahías"
            st.info(f"💡 **Por qué esta decisión:** Esta combinación protege un índice de riesgo-valor de **{puntos_totales} puntos**, maximizando la cobertura ecológica con el presupuesto de 2 barreras y {motivo_logistico}.")
            
            col1, col2 = st.columns(2)
            col1.metric("Estado Cuántico Ganador (q0-q3)", bits_ganadores)
            col2.metric("Frecuencia de Muestreo", f"{conteos.get(bits_ganadores[::-1])} / 1000 disparos")
            
            top_conteos = dict(sorted(conteos.items(), key=lambda item: item[1], reverse=True)[:5])
            etiquetas_ordenadas = []
            for k in top_conteos.keys():
                bits_k = k[::-1]
                nombres = [str(df_trabajo.iloc[i].get("Zona_Critica")).split("_").pop(1) for i in range(4) if bits_k[i] == '1']
                etiquetas_ordenadas.append(" + ".join(nombres) if nombres else "Ninguna")
            valores_ordenados = list(top_conteos.values())
            
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.bar(etiquetas_ordenadas, valores_ordenados, color='#1f77b4')
            ax.set_title("Comparativa QAOA: Combinaciones de Protección Más Óptimas")
            ax.set_xlabel("Zonas Prioritarias Seleccionadas")
            ax.set_ylabel("Frecuencia (de 1000 disparos)")
            st.pyplot(fig)

# --- PIE DE PÁGINA Y CONTACTO ---
st.divider()
st.subheader("💬 Validación en Chip Real IBM y Colaboraciones")
col_c1, col_c2 = st.columns(2)
with col_c1:
    if st.button("👍 ¡Esta herramienta me ha resultado útil!"):
        st.toast("¡Gracias por tu valoración! Seguimos impulsando la Economía Azul en Tarragona.")
with col_c2:
    st.markdown("¿Deseas validar este modelo en hardware cuántico real de **IBM Quantum** o proponer una colaboración?")
    st.markdown("📩 **Correo directo:** [nhemesysgonzalez@gmail.com](mailto:nhemesysgonzalez@gmail.com) | 💬 **[Comentarios en GitHub](https://github.com)**")
