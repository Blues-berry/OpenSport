from weak_session import evaluate_weak_session


def test_weak_session_reports_coverage_order_and_set_error():
    result = evaluate_weak_session(
        {
            "ordered_activities": [
                {"activity_id": "lat_pulldown"},
                {"activity_id": "bench_press"},
                {"activity_id": "run"},
            ],
            "total_sets": 5,
        },
        [
            {"activity_id": "lat_pulldown", "sets": 2},
            {"activity_id": "run", "sets": 0},
        ],
    )
    assert result["activity_coverage"] == 2 / 3
    assert result["order_score"] == 2 / 3
    assert result["set_count_error"] == -3
