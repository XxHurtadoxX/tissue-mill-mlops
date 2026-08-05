"""Verifica las definiciones de Azure Machine Learning antes de enviarlas.

La mitad de este proyecto son archivos YAML, y un error ahí no lo detecta ni el
lint de Python ni las pruebas del modelo: se descubre cuando el trabajo falla en
la nube, minutos después y habiendo consumido cómputo.

Ya ocurrió dos veces. Un carácter de dos puntos dentro de una descripción sin
comillas rompió el parseo de un componente, y un entorno curado resultó no
servir para una búsqueda de hiperparámetros. Ninguna de las dos cosas se habría
visto sin ejecutar.

Estas comprobaciones cuestan milisegundos y corren en cada cambio.
"""
import pathlib

import pytest
import yaml

RAIZ = pathlib.Path(__file__).resolve().parents[2]
AML = RAIZ / "aml"

ARCHIVOS = sorted(AML.rglob("*.yml"))


def _cargar(ruta: pathlib.Path) -> dict:
    with open(ruta, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_hay_definiciones_que_revisar():
    """Si esto falla, el resto de pruebas pasaría en vacío."""
    assert len(ARCHIVOS) >= 10


@pytest.mark.parametrize("ruta", ARCHIVOS, ids=lambda p: p.name)
def test_el_yaml_es_valido(ruta):
    """Un dos puntos sin comillas dentro de una descripción rompe el parseo."""
    contenido = _cargar(ruta)
    assert isinstance(contenido, dict), f"{ruta.name} no define un objeto"


@pytest.mark.parametrize("ruta", ARCHIVOS, ids=lambda p: p.name)
def test_las_rutas_relativas_existen(ruta):
    """Un `path` o un `code` que apunte a la nada falla recién en la nube."""
    contenido = _cargar(ruta)
    for clave in ("path", "code", "conda_file", "create_job"):
        valor = contenido.get(clave)
        if not isinstance(valor, str):
            continue
        # Las referencias a assets del workspace no son rutas de disco.
        if valor.startswith(("azureml:", "azureml://", "http")):
            continue
        assert (ruta.parent / valor).exists(), (
            f"{ruta.name} declara {clave}: {valor}, que no existe")


def test_el_sweep_busca_la_metrica_que_el_script_registra():
    """El fallo silencioso más caro de una búsqueda de hiperparámetros.

    Si el nombre no coincide, el sweep corre sin errores, termina, y devuelve
    los intentos sin ordenar porque no encontró ningún valor que comparar. Se
    descubre después de haber pagado cuarenta entrenamientos.
    """
    from src.model.train import METRICA_PRINCIPAL

    sweep = _cargar(AML / "jobs" / "sweep-job.yml")
    declarada = sweep["objective"]["primary_metric"]
    assert declarada == METRICA_PRINCIPAL, (
        f"El sweep busca '{declarada}' y el script registra "
        f"'{METRICA_PRINCIPAL}'. Con nombres distintos la búsqueda no puede "
        f"ordenar los intentos.")


def test_el_sweep_pasa_todos_los_parametros_que_recorre():
    """Un parámetro en el espacio de búsqueda que el comando no pasa se ignora."""
    sweep = _cargar(AML / "jobs" / "sweep-job.yml")
    comando = sweep["trial"]["command"]
    for parametro in sweep["search_space"]:
        assert f"search_space.{parametro}" in comando, (
            f"'{parametro}' está en el espacio de búsqueda pero el comando no "
            f"lo pasa al script: se recorrería sin efecto")


def test_los_trabajos_declaran_limites_de_tiempo():
    """Sin tope, un trabajo puede consumir el clúster durante horas."""
    for nombre in ("sweep-job.yml", "automl-classification.yml"):
        contenido = _cargar(AML / "jobs" / nombre)
        limites = contenido.get("limits", {})
        assert limites.get("timeout") or limites.get("timeout_minutes"), (
            f"{nombre} no declara tope de tiempo")


def test_el_cluster_puede_escalar_a_cero():
    """Con el mínimo en uno se paga un nodo las veinticuatro horas."""
    cluster = _cargar(AML / "setup" / "compute-cluster.yml")
    assert cluster["min_instances"] == 0


def test_los_componentes_no_fijan_su_computo():
    """Un componente que declara su máquina deja de ser reutilizable."""
    for ruta in sorted((AML / "components").glob("*.yml")):
        contenido = _cargar(ruta)
        assert "compute" not in contenido, (
            f"{ruta.name} fija su cómputo: eso se decide en el pipeline")
