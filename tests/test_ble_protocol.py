import pytest

from ble_protocol import CompactSample, SAMPLE_STRUCT, decode_batch, encode_batch


def test_compact_sample_is_fourteen_bytes_and_round_trips():
    source = [
        CompactSample(41, 0.25, -0.5, 1.0, 12.0, -40.0, 90.0),
        CompactSample(42, 0.1, 0.2, 0.9, -10.0, 20.0, -30.0),
    ]
    packet = encode_batch(source)
    decoded = decode_batch(packet)
    assert SAMPLE_STRUCT.size == 14
    assert [item.sequence_id for item in decoded] == [41, 42]
    assert decoded[0].az_g == pytest.approx(1.0, abs=0.001)
    assert decoded[0].gy_dps == pytest.approx(-40.0, abs=0.1)


def test_crc_rejects_corruption():
    packet = bytearray(encode_batch([CompactSample(1, 0, 0, 1, 0, 0, 0)]))
    packet[8] ^= 0x01
    with pytest.raises(ValueError, match="CRC"):
        decode_batch(bytes(packet))
