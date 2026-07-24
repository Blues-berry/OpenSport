"""Compact 50 Hz BLE payload: sequence id plus six signed 16-bit axes."""

from __future__ import annotations

import struct
from dataclasses import dataclass


MAGIC = b"IM"
VERSION = 1
SAMPLE_STRUCT = struct.Struct("<H6h")
HEADER_STRUCT = struct.Struct("<2sBB")
CRC_STRUCT = struct.Struct("<H")
ACCEL_SCALE_G = 16.0 / 32768.0
GYRO_SCALE_DPS = 2000.0 / 32768.0


@dataclass(frozen=True)
class CompactSample:
    sequence_id: int
    ax_g: float
    ay_g: float
    az_g: float
    gx_dps: float
    gy_dps: float
    gz_dps: float

    def runtime_dict(self, timestamp: float) -> dict:
        return {
            "timestamp": timestamp,
            "sequence_id": self.sequence_id,
            "ax_g": self.ax_g,
            "ay_g": self.ay_g,
            "az_g": self.az_g,
            "gx_dps": self.gx_dps,
            "gy_dps": self.gy_dps,
            "gz_dps": self.gz_dps,
        }


def crc16_ccitt(data: bytes, initial: int = 0xFFFF) -> int:
    crc = initial
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _quantize(value: float, full_scale: float) -> int:
    return max(-32768, min(32767, int(round(value / full_scale * 32768.0))))


def encode_sample(sample: CompactSample) -> bytes:
    return SAMPLE_STRUCT.pack(
        sample.sequence_id & 0xFFFF,
        _quantize(sample.ax_g, 16.0),
        _quantize(sample.ay_g, 16.0),
        _quantize(sample.az_g, 16.0),
        _quantize(sample.gx_dps, 2000.0),
        _quantize(sample.gy_dps, 2000.0),
        _quantize(sample.gz_dps, 2000.0),
    )


def decode_sample(payload: bytes) -> CompactSample:
    if len(payload) != SAMPLE_STRUCT.size:
        raise ValueError(f"Compact sample must be {SAMPLE_STRUCT.size} bytes")
    sequence, ax, ay, az, gx, gy, gz = SAMPLE_STRUCT.unpack(payload)
    return CompactSample(
        sequence,
        ax * ACCEL_SCALE_G,
        ay * ACCEL_SCALE_G,
        az * ACCEL_SCALE_G,
        gx * GYRO_SCALE_DPS,
        gy * GYRO_SCALE_DPS,
        gz * GYRO_SCALE_DPS,
    )


def encode_batch(samples: list[CompactSample]) -> bytes:
    if not 1 <= len(samples) <= 15:
        raise ValueError("A notification batch must contain 1 to 15 samples")
    body = HEADER_STRUCT.pack(MAGIC, VERSION, len(samples)) + b"".join(encode_sample(sample) for sample in samples)
    return body + CRC_STRUCT.pack(crc16_ccitt(body))


def decode_batch(packet: bytes) -> list[CompactSample]:
    if len(packet) < HEADER_STRUCT.size + CRC_STRUCT.size:
        raise ValueError("Packet is too short")
    magic, version, count = HEADER_STRUCT.unpack_from(packet)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported compact IMU packet")
    expected = HEADER_STRUCT.size + count * SAMPLE_STRUCT.size + CRC_STRUCT.size
    if len(packet) != expected:
        raise ValueError(f"Packet length mismatch: expected {expected}, got {len(packet)}")
    expected_crc = CRC_STRUCT.unpack_from(packet, len(packet) - CRC_STRUCT.size)[0]
    if crc16_ccitt(packet[:-CRC_STRUCT.size]) != expected_crc:
        raise ValueError("CRC check failed")
    offset = HEADER_STRUCT.size
    return [
        decode_sample(packet[offset + index * SAMPLE_STRUCT.size : offset + (index + 1) * SAMPLE_STRUCT.size])
        for index in range(count)
    ]
