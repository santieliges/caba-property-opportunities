# Updater

## Proposito

Esta carpeta encapsula el refresh de un aviso ya conocido. Su rol es tomar un
id y una fila activa, consultar la fuente remota y devolver la representacion
normalizada del aviso.

## Archivo principal

- [updater.py](/home/saneliges/Escritorio/caba-property-opportunities/scraper_service/updater/updater.py)
  Define `Updater`.

## Flujo

`Updater.fetch(entry_id, entry, argenprop_scraper)`:

1. toma la URL del aviso desde la fila activa
2. intenta leer detalle desde Sosiva
3. si Sosiva responde `200`, mapea la respuesta a `InmuebleData`
4. si Sosiva responde `404` o `410`, devuelve `410`
5. en cualquier otro caso, hace fallback al scraping HTML del detalle

## Inputs

- `entry_id`
  Id de Argenprop.
- `entry`
  Fila actual del aviso en storage.
- `argenprop_scraper`
  Instancia viva del scraper para hacer fallback al HTML.

## Outputs

- `dict`
  Aviso normalizado y listo para sincronizar.
- `410`
  El aviso debe considerarse cerrado/inactivo.
- `None`
  No hay URL util para refrescar el aviso.

## Alcance

Editar esta carpeta cuando cambie:

- la prioridad entre API y scraping HTML
- la interpretacion de codigos HTTP remotos
- el contrato de salida del refresh individual

No editar esta carpeta para:

- rediseñar el pipeline entero
- tocar storage o versionado historico
- manejar navegacion de Playwright en general
