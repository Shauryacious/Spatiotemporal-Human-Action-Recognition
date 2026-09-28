def observed_boxes(report, timestamp: float, tolerance: float = 0.125):
    """Sparse evidence only; never imply boxes exist throughout an unobserved gap."""
    return [
        {"actor_id": track.actor_id, **sample.model_dump(mode="json")}
        for track in report.tracks
        for sample in track.samples
        if sample.observed and abs(sample.timestamp_sec - timestamp) <= tolerance
    ]


def timeline(report):
    return [
        {
            "id": p.id,
            "label": p.label,
            "start_sec": p.start_sec,
            "end_sec": p.end_sec,
            "confidence": p.confidence,
            "spatial_status": p.spatial_status,
        }
        for p in report.predictions
    ]
