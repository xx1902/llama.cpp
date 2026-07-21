#!/usr/bin/env python3
"""Online per-user GRU service for the lora-base-test2 C++ experiment.

The service owns prediction state only. llama.cpp remains responsible for
prefix matching, anchor-family scoring, KV reconstruction, and prefetching.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class TinyGRUPredictor(nn.Module):
    """Architecture used by base_model_global.pth from the original notebook."""

    def __init__(
        self,
        num_apps: int = 87,
        embed_dim: int = 8,
        hidden_dim: int = 64,
        time_dim: int = 3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_apps, embed_dim)
        self.gru = nn.GRU(embed_dim + time_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_apps)

    def forward(self, app_ids: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(app_ids)
        output, _ = self.gru(torch.cat([embedded, time_features], dim=-1))
        return self.fc(output[:, -1, :])


@dataclass
class UserState:
    app_ids: deque[int]
    time_features: deque[list[float]]
    last_timestamp: datetime | None = None
    last_request_id: int = -1
    events_seen: int = 0
    last_response: dict[str, Any] | None = None


def load_mapping(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    lora_to_app: dict[int, int] = {}
    app_to_lora: dict[int, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            app_id = int(row["app_id"])
            lora_id = int(row["lora_id"])
            if app_id in app_to_lora or lora_id in lora_to_app:
                raise ValueError("App/LoRA mapping must be one-to-one")
            app_to_lora[app_id] = lora_id
            lora_to_app[lora_id] = app_id
    if len(app_to_lora) != 87:
        raise ValueError(f"expected 87 App/LoRA mappings, got {len(app_to_lora)}")
    if set(app_to_lora) != set(range(87)):
        raise ValueError("app_id values must be the LabelEncoder IDs 0..86")
    return lora_to_app, app_to_lora


def parse_timestamp(payload: dict[str, Any]) -> datetime:
    original = str(payload.get("original_timestamp") or "").strip()
    if original:
        return datetime.fromisoformat(original.replace("Z", "+00:00"))
    return datetime.fromtimestamp(float(payload.get("arrival_ms", 0)) / 1000.0)


class OnlineGruPredictor:
    def __init__(
        self,
        checkpoint: Path,
        mapping: Path,
        window: int,
        min_history: int,
        device_name: str,
        log_file: Path | None,
    ) -> None:
        self.window = window
        self.min_history = min_history
        self.device = torch.device(
            device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu"
        )
        self.lora_to_app, self.app_to_lora = load_mapping(mapping)
        self.model = TinyGRUPredictor(num_apps=len(self.app_to_lora)).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()
        # Pay one-time PyTorch kernel initialization before serving requests.
        with torch.inference_mode():
            warmup_ids = torch.zeros((1, window), dtype=torch.long, device=self.device)
            warmup_time = torch.zeros((1, window, 3), dtype=torch.float32, device=self.device)
            for _ in range(3):
                self.model(warmup_ids, warmup_time)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.states: dict[int, UserState] = {}
        self.lock = threading.Lock()
        self.log_file = log_file
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> int:
        with self.lock:
            users = len(self.states)
            self.states.clear()
            return users

    def state_summary(self) -> dict[str, Any]:
        with self.lock:
            return {
                "users": len(self.states),
                "events": sum(state.events_seen for state in self.states.values()),
                "window": self.window,
                "min_history": self.min_history,
                "device": str(self.device),
            }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = int(payload["request_id"])
        user_id = int(payload["user_id"])
        lora_id = int(payload["lora_id"])
        top_k = max(1, min(int(payload.get("top_k", 3)), len(self.app_to_lora)))
        if user_id < 0:
            raise ValueError("user_id must be non-negative")
        if lora_id not in self.lora_to_app:
            raise ValueError(f"unknown lora_id={lora_id}")

        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            state = self.states.get(user_id)
            if state is None:
                state = UserState(
                    app_ids=deque(maxlen=self.window),
                    time_features=deque(maxlen=self.window),
                )
                self.states[user_id] = state

            # HTTP retries must not append the same observation twice.
            if request_id == state.last_request_id and state.last_response is not None:
                response = dict(state.last_response)
                response["duplicate"] = True
                return response
            if request_id < state.last_request_id:
                raise ValueError(
                    f"out-of-order request for user {user_id}: "
                    f"current={request_id}, last={state.last_request_id}"
                )

            current_time = parse_timestamp(payload)
            delta_seconds = 0.0
            if state.last_timestamp is not None:
                delta_seconds = max(0.0, (current_time - state.last_timestamp).total_seconds())
            hour = current_time.hour
            features = [
                math.sin(2.0 * math.pi * hour / 24.0),
                math.cos(2.0 * math.pi * hour / 24.0),
                math.log1p(delta_seconds),
            ]
            state.app_ids.append(self.lora_to_app[lora_id])
            state.time_features.append(features)
            state.last_timestamp = current_time
            state.last_request_id = request_id
            state.events_seen += 1

            response: dict[str, Any] = {
                "status": "ok",
                "request_id": request_id,
                "user_id": user_id,
                "ready": False,
                "history_length": len(state.app_ids),
                "events_seen": state.events_seen,
                "inference_ms": 0.0,
                "predictions": [],
                "duplicate": False,
            }
            if len(state.app_ids) >= self.min_history:
                ids = torch.tensor(
                    [list(state.app_ids)], dtype=torch.long, device=self.device
                )
                times = torch.tensor(
                    [list(state.time_features)], dtype=torch.float32, device=self.device
                )
                inference_start = time.perf_counter()
                probabilities = torch.softmax(self.model(ids, times), dim=1)[0]
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                response["inference_ms"] = (
                    time.perf_counter() - inference_start
                ) * 1000.0
                values, indices = torch.topk(probabilities, k=top_k)
                response["predictions"] = [
                    {
                        "app_id": int(app_id),
                        "lora_id": self.app_to_lora[int(app_id)],
                        "probability": float(probability),
                    }
                    for app_id, probability in zip(
                        indices.cpu().tolist(), values.cpu().tolist()
                    )
                ]
                response["ready"] = True

            response["service_ms"] = (time.perf_counter() - started) * 1000.0
            state.last_response = dict(response)
            if self.log_file is not None:
                with self.log_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(response, ensure_ascii=False) + "\n")
            return response


class PredictionHandler(BaseHTTPRequestHandler):
    predictor: OnlineGruPredictor

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1024 * 1024:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"status": "ok", **self.predictor.state_summary()})
        elif self.path == "/state":
            self.send_json(200, self.predictor.state_summary())
        else:
            self.send_json(404, {"status": "error", "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.read_json()
            if self.path == "/predict":
                self.send_json(200, self.predictor.predict(payload))
            elif self.path == "/reset":
                users = self.predictor.reset()
                self.send_json(200, {"status": "ok", "cleared_users": users})
            else:
                self.send_json(404, {"status": "error", "error": "not found"})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_json(400, {"status": "error", "error": str(error)})
        except Exception as error:  # Keep the service alive and expose the failure.
            self.send_json(500, {"status": "error", "error": str(error)})

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            r"D:\ecnu_experiment\LoRA\clustering_weights\user_Subsequence_exponential411_time10\base_model_global.pth"
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            r"D:\ecnu_experiment\datasets\mobilora_workloads_87_original\app_lora_mapping.csv"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--min-history", type=int, default=15)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    if args.window <= 0 or not 1 <= args.min_history <= args.window:
        parser.error("require 1 <= min-history <= window")

    torch.set_num_threads(max(1, args.torch_threads))
    PredictionHandler.predictor = OnlineGruPredictor(
        checkpoint=args.checkpoint,
        mapping=args.mapping,
        window=args.window,
        min_history=args.min_history,
        device_name=args.device,
        log_file=args.log_file,
    )
    server = ThreadingHTTPServer((args.host, args.port), PredictionHandler)
    print(
        f"GRU prediction service: http://{args.host}:{args.port} "
        f"device={PredictionHandler.predictor.device} window={args.window}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
