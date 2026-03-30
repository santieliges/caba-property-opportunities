import argparse
import json
import os
from typing import Any, Dict, Optional

import requests


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_headers(path: str) -> Dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Headers JSON debe ser un objeto {\"Header\": \"valor\", ...}")
    return {str(k): str(v) for k, v in data.items()}


def dump_response_text(
    *,
    out_path: str,
    url: str,
    method: str,
    request_headers: Dict[str, str],
    request_json: Optional[Any],
    status_code: int,
    response_headers: Dict[str, str],
    response_text: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"method: {method}\n")
        f.write(f"url: {url}\n")
        f.write(f"status_code: {status_code}\n")
        f.write("\nrequest_headers:\n")
        f.write(json.dumps(request_headers, ensure_ascii=False, indent=2))
        f.write("\n\nrequest_json:\n")
        f.write(json.dumps(request_json, ensure_ascii=False, indent=2) if request_json is not None else "null")
        f.write("\n\nresponse_headers:\n")
        f.write(json.dumps(response_headers, ensure_ascii=False, indent=2))
        f.write("\n\nresponse_body:\n")
        f.write(response_text)
        f.write("\n")


def dump_response_json(
    *,
    out_path: str,
    url: str,
    method: str,
    request_headers: Dict[str, str],
    request_json: Optional[Any],
    status_code: int,
    response_headers: Dict[str, str],
    response_json: Any,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "method": method,
        "url": url,
        "status_code": status_code,
        "request_headers": request_headers,
        "request_json": request_json,
        "response_headers": response_headers,
        "response_json": response_json,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Testea https://www.argenprop.com/ignored/IsIgnored (requiere headers/cookies según el sitio)."
    )
    parser.add_argument(
        "--url",
        default="https://www.argenprop.com/ignored/IsIgnored",
        help="URL del endpoint a probar.",
    )
    parser.add_argument(
        "--method",
        default="GET",
        choices=["GET", "POST"],
        help="Método HTTP.",
    )
    parser.add_argument(
        "--headers",
        help="Path a JSON con headers (ej: Cookie, User-Agent). Alternativa: env ARGENPROP_HEADERS_JSON.",
    )
    parser.add_argument(
        "--data",
        help="Path a JSON con el body (para POST). Alternativa: env ARGENPROP_DATA_JSON.",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("ARGENPROP_OUT", "output/isignored_test.txt"),
        help="Path donde guardar el resultado (.txt o .json). Alternativa: env ARGENPROP_OUT.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Timeout en segundos.",
    )
    args = parser.parse_args()

    headers_path = args.headers or os.environ.get("ARGENPROP_HEADERS_JSON")
    data_path = args.data or os.environ.get("ARGENPROP_DATA_JSON")

    headers: Dict[str, str] = load_headers(headers_path) if headers_path else {}
    payload = load_json(data_path) if data_path else None

    # Defaults mínimos para que el endpoint no se caiga por headers vacíos.
    headers.setdefault("Accept", "application/json, text/plain, */*")
    headers.setdefault("User-Agent", "Mozilla/5.0")

    method = args.method.upper()
    if method == "GET":
        r = requests.get(args.url, headers=headers, timeout=args.timeout)
    else:
        r = requests.post(args.url, headers=headers, json=payload, timeout=args.timeout)

    print("status_code:", r.status_code)
    print("content_type:", r.headers.get("content-type"))
    print("out:", args.out)

    # Dump: si parece JSON, guardarlo como JSON si el out termina en .json; si no, texto.
    out_ext = os.path.splitext(args.out)[1].lower()
    response_headers = {k: v for k, v in r.headers.items()}

    if out_ext == ".json":
        try:
            dump_response_json(
                out_path=args.out,
                url=args.url,
                method=method,
                request_headers=headers,
                request_json=payload,
                status_code=r.status_code,
                response_headers=response_headers,
                response_json=r.json(),
            )
        except Exception:
            dump_response_text(
                out_path=args.out,
                url=args.url,
                method=method,
                request_headers=headers,
                request_json=payload,
                status_code=r.status_code,
                response_headers=response_headers,
                response_text=r.text,
            )
    else:
        dump_response_text(
            out_path=args.out,
            url=args.url,
            method=method,
            request_headers=headers,
            request_json=payload,
            status_code=r.status_code,
            response_headers=response_headers,
            response_text=r.text,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

