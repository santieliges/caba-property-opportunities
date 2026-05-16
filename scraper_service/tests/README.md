# Tests

## Proposito

Esta carpeta contiene pruebas automatizadas y algunos checks de integracion del
scraper. Sirve para validar heuristicas de negocio y partes del flujo de update.

## Archivos actuales

- [test_pozo_heuristic.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/tests/test_pozo_heuristic.py)
  Valida la heuristica de deteccion de propiedades en pozo y parte del mapping
  desde Sosiva.
- [test_single_update_flow.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/tests/test_single_update_flow.py)
  Ejecuta un flujo acotado de update de un aviso real.
- [example.spec.ts](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/tests/example.spec.ts)
  Ejemplo base de Playwright. No cubre el flujo principal de negocio.

## Alcance

Editar esta carpeta cuando cambie:

- una heuristica de negocio como `pozo`
- el contrato de `Updater` o `RoutineJob`
- la forma de integrar scraping y sincronizacion sobre un caso real acotado

No editar esta carpeta para:

- agregar documentacion operativa
- arreglar wrappers Docker

## Cobertura y limites

- Hay foco en heuristicas puntuales y smoke coverage del update.
- No hay una suite amplia de pruebas unitarias por capa.
- Algunas pruebas dependen de servicios externos y pueden volverse inestables si
  cambia Argenprop o Sosiva.
