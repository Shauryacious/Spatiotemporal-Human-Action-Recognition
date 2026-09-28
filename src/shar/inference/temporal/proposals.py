def candidates(payload, threshold: float, cap: int):
    return [p for p in payload.predictions if p.confidence is None or p.confidence >= threshold][
        :cap
    ]
