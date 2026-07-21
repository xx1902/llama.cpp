#!/usr/bin/env python3
"""State-management smoke test for gru_online_service.py."""

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from gru_online_service import (
    OnlineGruPredictor,
    PredictionHandler,
    ThreadingHTTPServer,
)


CHECKPOINT = Path(
    r"D:\ecnu_experiment\LoRA\clustering_weights\user_Subsequence_exponential411_time10\base_model_global.pth"
)
MAPPING = Path(
    r"D:\ecnu_experiment\datasets\mobilora_workloads_87_original\app_lora_mapping.csv"
)


def event(request_id: int, user_id: int, lora_id: int) -> dict:
    return {
        "request_id": request_id,
        "user_id": user_id,
        "lora_id": lora_id,
        "original_timestamp": f"2018-01-04T05:{request_id % 60:02d}:00",
        "top_k": 3,
    }


def post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    torch.set_num_threads(1)
    predictor = OnlineGruPredictor(
        checkpoint=CHECKPOINT,
        mapping=MAPPING,
        window=15,
        min_history=15,
        device_name="cpu",
        log_file=None,
    )

    responses = [predictor.predict(event(index, 138, index)) for index in range(15)]
    assert all(not row["ready"] for row in responses[:14])
    assert responses[14]["ready"]
    assert len(responses[14]["predictions"]) == 3

    duplicate = predictor.predict(event(14, 138, 14))
    assert duplicate["duplicate"]
    assert duplicate["events_seen"] == 15

    other_user = predictor.predict(event(100, 999, 0))
    assert not other_user["ready"]
    assert other_user["events_seen"] == 1
    assert predictor.state_summary()["users"] == 2

    PredictionHandler.predictor = predictor
    server = ThreadingHTTPServer(("127.0.0.1", 0), PredictionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(f"{base_url}/health", timeout=5) as response:
            assert json.loads(response.read().decode("utf-8"))["status"] == "ok"
        assert post_json(f"{base_url}/reset", {})["status"] == "ok"
        http_responses = [
            post_json(f"{base_url}/predict", event(index, 138, index))
            for index in range(15)
        ]
        assert not http_responses[13]["ready"]
        assert http_responses[14]["ready"]
        assert len(http_responses[14]["predictions"]) == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    print("GRU online service smoke test: PASS")
    print(
        f"top1_lora={responses[14]['predictions'][0]['lora_id']} "
        f"inference_ms={responses[14]['inference_ms']:.3f}"
    )


if __name__ == "__main__":
    main()
