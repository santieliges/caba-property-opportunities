import argparse
import os
import json

from scraper_service.scrapper.SosivaApiClient import (
    SosivaApiClient,
    load_headers_from_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Testea https://api.sosiva451.com/Avisos/{id}")
    parser.add_argument("aviso_id", type=int, help="ID del aviso (entero)")
    parser.add_argument(
        "--headers",
        help="Path a un JSON con headers (por ejemplo Authorization/Cookie). Alternativa: env SOSIVA_HEADERS_JSON.",
    )
    parser.add_argument(
        "--out",
        help="Path donde guardar el resultado (txt o json). Alternativa: env SOSIVA_OUT.",
    )
    args = parser.parse_args()

    headers_path = args.headers or os.environ.get("SOSIVA_HEADERS_JSON")
    headers = load_headers_from_json(headers_path) if headers_path else None

    client = SosivaApiClient(headers=headers)
    res = client.get_aviso(args.aviso_id)

    out_path = args.out or os.environ.get("SOSIVA_OUT")
    payload = {
        "status_code": res.status_code,
        "url": res.url,
        "json": res.json_data,
        "text": res.text,
    }

    # Siempre imprimir un resumen corto en stdout.
    print("status_code:", res.status_code)
    print("url:", res.url)
    if res.json_data is not None:
        print("json_keys:", list(res.json_data.keys())[:50])
    else:
        print("text_prefix:", (res.text or "")[:500])

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        ext = os.path.splitext(out_path)[1].lower()
        if ext == ".json":
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(f"status_code: {res.status_code}\n")
                f.write(f"url: {res.url}\n")
                if res.json_data is not None:
                    f.write("json:\n")
                    f.write(json.dumps(res.json_data, ensure_ascii=False, indent=2))
                    f.write("\n")
                else:
                    f.write("text:\n")
                    f.write((res.text or "")[:20000])
                    f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
