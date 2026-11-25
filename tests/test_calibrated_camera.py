from src.core.calibrated_camera import Region, decide


def region(x, y):
    return Region((x, y), (x - 5, y - 5, x + 5, y + 5), 4., 120)


def test_borderline_capture_waits_for_confirmation_and_filters_regions():
    first = decide(6.1, 6., (region(100, 100),))
    assert first.action == "CONFIRM"
    assert not first.regions
    next_frame = (region(108, 108), region(200, 200))
    confirmed = decide(5.5, 6., next_frame, first.pending)
    assert confirmed.action == "INSPECT"
    assert confirmed.regions == (next_frame[0],)
    assert confirmed.pending is None


def test_unusable_confirmation_preserves_pending_capture():
    first = decide(8., 6., (region(100, 100),))
    for seed in (None, float("nan"), 9.1):
        result = decide(seed, 6., (), first.pending)
        assert result.action == "RECAPTURE"
        assert result.pending == first.pending


def test_a_region_in_the_first_capture_cannot_confirm_two_regions():
    first = decide(6.1, 6., (region(100, 100),))
    closest = region(103, 103)
    assert decide(6., 6., (region(106, 106), closest), first.pending).regions == (closest,)
    assert decide(6., 6., (), first.pending).pending is None

