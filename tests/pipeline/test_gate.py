"""Pruebas de la compuerta que decide si reemplazar el modelo de producción.

Es la lógica más delicada del pipeline. Un error aquí no rompe nada de forma
visible: simplemente reemplaza un modelo bueno por uno peor, y nadie se entera
hasta que mantenimiento deja de recibir alertas útiles.
"""
import numpy as np

from src.model.gate import AUC_MINIMO, decidir

CAPACIDAD = 8


def _metricas(auc=0.90, detectables=4, dias=5.0, alertas=8.0):
    return {
        "auc": auc,
        "detectables_anticipados": detectables,
        "eventos_detectados": detectables,
        "anticipacion_dias": dias,
        "alertas_mes": alertas,
    }


# --------------------------------------------------------------------------- #
# El caso que da sentido a la compuerta
# --------------------------------------------------------------------------- #
def test_un_modelo_peor_no_se_registra():
    registrar, razon = decidir(
        _metricas(detectables=2), _metricas(detectables=4), CAPACIDAD)
    assert registrar is False
    assert "conserva" in razon.lower()


def test_un_modelo_que_anticipa_mas_eventos_si_se_registra():
    registrar, _ = decidir(
        _metricas(detectables=5), _metricas(detectables=4), CAPACIDAD)
    assert registrar is True


def test_el_primer_modelo_siempre_se_registra():
    registrar, razon = decidir(_metricas(), None, CAPACIDAD)
    assert registrar is True
    assert "producción" in razon


# --------------------------------------------------------------------------- #
# El AUC no decide
# --------------------------------------------------------------------------- #
def test_mas_auc_no_basta_si_anticipa_menos_eventos():
    """Un modelo puede ganar AUC sin anticipar una sola falla más."""
    candidato = _metricas(auc=0.95, detectables=3)
    produccion = _metricas(auc=0.85, detectables=4)
    registrar, _ = decidir(candidato, produccion, CAPACIDAD)
    assert registrar is False


def test_menos_auc_no_impide_registrar_si_anticipa_mas_eventos():
    candidato = _metricas(auc=0.86, detectables=5)
    produccion = _metricas(auc=0.93, detectables=4)
    registrar, _ = decidir(candidato, produccion, CAPACIDAD)
    assert registrar is True


# --------------------------------------------------------------------------- #
# Desempate por antelación
# --------------------------------------------------------------------------- #
def test_a_igual_numero_de_eventos_gana_el_que_avisa_antes():
    """Avisar dos días antes no alcanza para conseguir el repuesto."""
    registrar, _ = decidir(
        _metricas(dias=9.0), _metricas(dias=5.0), CAPACIDAD)
    assert registrar is True


def test_a_igual_numero_de_eventos_pierde_el_que_avisa_mas_tarde():
    registrar, _ = decidir(
        _metricas(dias=3.0), _metricas(dias=7.0), CAPACIDAD)
    assert registrar is False


def test_el_empate_conserva_el_modelo_ya_validado_en_planta():
    registrar, razon = decidir(_metricas(), _metricas(), CAPACIDAD)
    assert registrar is False
    assert "sin mejora" in razon.lower()


# --------------------------------------------------------------------------- #
# Salvaguardas
# --------------------------------------------------------------------------- #
def test_un_desplome_de_auc_frena_el_registro():
    """Suele significar que los datos cambiaron, no que el modelo mejoró."""
    candidato = _metricas(auc=AUC_MINIMO - 0.05, detectables=9)
    registrar, razon = decidir(candidato, _metricas(detectables=4), CAPACIDAD)
    assert registrar is False
    assert "piso" in razon.lower()


def test_no_se_registra_un_modelo_que_desborda_la_capacidad():
    """Inundar de alertas destruye la confianza y con ella el valor del sistema."""
    candidato = _metricas(detectables=6, alertas=CAPACIDAD * 3)
    registrar, razon = decidir(candidato, _metricas(detectables=4), CAPACIDAD)
    assert registrar is False
    assert "alertas" in razon.lower()


def test_funciona_sin_datos_de_anticipacion():
    """Si falta el dato de antelación, el empate no debe romper la decisión."""
    candidato = _metricas(dias=np.nan)
    produccion = _metricas(dias=np.nan)
    registrar, _ = decidir(candidato, produccion, CAPACIDAD)
    assert registrar is False
