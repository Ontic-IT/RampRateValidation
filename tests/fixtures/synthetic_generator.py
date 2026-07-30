"""Synthetic trace generator for calibration fixtures.

Generates deterministic synthetic traces using numpy with ALGORITHM_RANDOM_SEED.
All traces are fully reproducible.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from config.algorithm_versions import ALGORITHM_RANDOM_SEED


def generate_all_phase2_fixtures(output_dir: str | Path) -> list[str]:
    """Generate all 8 Phase 2 synthetic fixtures.
    
    Returns list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generators = [
        ("syn_clean_heating_ramp.csv", generate_clean_heating_ramp),
        ("syn_noisy_heating_ramp.csv", generate_noisy_heating_ramp),
        ("syn_clean_cooling_ramp.csv", generate_clean_cooling_ramp),
        ("syn_hot_overshoot.csv", generate_hot_overshoot),
        ("syn_cold_overshoot.csv", generate_cold_overshoot),
        ("syn_ramp_taper.csv", generate_ramp_taper),
        ("syn_ramp_jitter.csv", generate_ramp_jitter),
        ("syn_partial_dwell.csv", generate_partial_dwell),
    ]
    
    generated = []
    for filename, generator in generators:
        filepath = output_dir / filename
        generator(filepath)
        generated.append(str(filepath))
    
    return generated


def _write_csv(filepath: Path, timestamps: list[datetime], temperatures: list[float]) -> None:
    """Write trace data to CSV file."""
    with open(filepath, "w") as f:
        f.write("timestamp,temperature_c\n")
        for ts, temp in zip(timestamps, temperatures):
            f.write(f"{ts.isoformat()},{temp:.4f}\n")


def _generate_timestamps(n_points: int, interval_seconds: float = 1.0) -> list[datetime]:
    """Generate evenly spaced timestamps."""
    base = datetime(2025, 1, 1, 0, 0, 0)
    return [base + timedelta(seconds=i * interval_seconds) for i in range(n_points)]


def generate_clean_heating_ramp(filepath: Path) -> None:
    """Linear heating ramp, no noise.
    
    Profile: 25°C ambient -> 5°C/min ramp -> 125°C dwell
    """
    np.random.seed(ALGORITHM_RANDOM_SEED)
    
    ambient_duration = 60
    ramp_duration = 1200
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    
    n_ambient = ambient_duration
    n_ramp = ramp_duration
    n_dwell = dwell_duration
    
    temps = []
    
    temps.extend([ambient_temp] * n_ambient)
    
    for i in range(n_ramp):
        t = ambient_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * n_dwell)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_noisy_heating_ramp(filepath: Path) -> None:
    """Heating ramp with Gaussian noise at 3× noise floor.
    
    Noise floor assumed ~0.1°C, so noise std = 0.3°C
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 1)
    
    ambient_duration = 60
    ramp_duration = 1200
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    noise_std = 0.3
    
    temps = []
    
    for _ in range(ambient_duration):
        temps.append(ambient_temp + np.random.normal(0, noise_std))
    
    for i in range(ramp_duration):
        t = ambient_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t + np.random.normal(0, noise_std))
    
    for _ in range(dwell_duration):
        temps.append(target_temp + np.random.normal(0, noise_std))
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_clean_cooling_ramp(filepath: Path) -> None:
    """Linear cooling ramp, no noise.
    
    Profile: 125°C dwell -> 5°C/min cooling -> -40°C dwell
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 2)
    
    dwell_duration = 300
    ramp_duration = 1980
    final_dwell_duration = 300
    
    start_temp = 125.0
    target_temp = -40.0
    ramp_rate = 5.0
    
    temps = []
    
    temps.extend([start_temp] * dwell_duration)
    
    for i in range(ramp_duration):
        t = start_temp - (ramp_rate / 60.0) * i
        t = max(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * final_dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_hot_overshoot(filepath: Path) -> None:
    """Heating ramp with overshoot above setpoint.
    
    Profile: 25°C -> ramp -> overshoot to 130°C -> settle to 125°C
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 3)
    
    ambient_duration = 60
    ramp_duration = 1200
    overshoot_duration = 30
    settling_duration = 60
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    overshoot_peak = 130.0
    ramp_rate = 5.0
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for i in range(ramp_duration):
        t = ambient_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    for i in range(overshoot_duration):
        progress = i / overshoot_duration
        t = target_temp + (overshoot_peak - target_temp) * np.sin(progress * np.pi / 2)
        temps.append(t)
    
    for i in range(settling_duration):
        progress = i / settling_duration
        t = overshoot_peak - (overshoot_peak - target_temp) * progress
        temps.append(t)
    
    temps.extend([target_temp] * dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_cold_overshoot(filepath: Path) -> None:
    """Cooling ramp with overshoot below setpoint.
    
    Profile: 125°C -> ramp -> overshoot to -45°C -> settle to -40°C
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 4)
    
    dwell_duration = 300
    ramp_duration = 1980
    overshoot_duration = 30
    settling_duration = 60
    final_dwell_duration = 300
    
    start_temp = 125.0
    target_temp = -40.0
    overshoot_trough = -45.0
    ramp_rate = 5.0
    
    temps = []
    
    temps.extend([start_temp] * dwell_duration)
    
    for i in range(ramp_duration):
        t = start_temp - (ramp_rate / 60.0) * i
        t = max(t, target_temp)
        temps.append(t)
    
    for i in range(overshoot_duration):
        progress = i / overshoot_duration
        t = target_temp + (overshoot_trough - target_temp) * np.sin(progress * np.pi / 2)
        temps.append(t)
    
    for i in range(settling_duration):
        progress = i / settling_duration
        t = overshoot_trough - (overshoot_trough - target_temp) * progress
        temps.append(t)
    
    temps.extend([target_temp] * final_dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_ramp_taper(filepath: Path) -> None:
    """Ramp with decelerating slope in final third.
    
    Profile: 25°C -> fast ramp -> tapered ramp -> 125°C
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 5)
    
    ambient_duration = 60
    fast_ramp_duration = 800
    taper_duration = 400
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    fast_rate = 6.0
    taper_rate = 2.0
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    current_temp = ambient_temp
    for i in range(fast_ramp_duration):
        current_temp = ambient_temp + (fast_rate / 60.0) * i
        current_temp = min(current_temp, target_temp)
        temps.append(current_temp)
    
    taper_start = current_temp
    for i in range(taper_duration):
        t = taper_start + (taper_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_ramp_jitter(filepath: Path) -> None:
    """Ramp with high-frequency noise throughout.
    
    Simulates controller oscillation during ramp.
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 6)
    
    ambient_duration = 60
    ramp_duration = 1200
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    jitter_amplitude = 1.0
    jitter_frequency = 0.1
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for i in range(ramp_duration):
        base_temp = ambient_temp + (ramp_rate / 60.0) * i
        base_temp = min(base_temp, target_temp)
        jitter = jitter_amplitude * np.sin(2 * np.pi * jitter_frequency * i)
        temps.append(base_temp + jitter)
    
    temps.extend([target_temp] * dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_partial_dwell(filepath: Path) -> None:
    """Dwell that doesn't reach minimum duration.
    
    Profile: 25°C -> ramp -> short 60s dwell (below 300s minimum)
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 7)
    
    ambient_duration = 60
    ramp_duration = 1200
    partial_dwell_duration = 60
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for i in range(ramp_duration):
        t = ambient_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * partial_dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_all_phase3_fixtures(output_dir: str | Path) -> list[str]:
    """Generate all 4 Phase 3 edge case synthetic fixtures.
    
    Returns list of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generators = [
        ("syn_sensor_dropout.csv", generate_sensor_dropout),
        ("syn_irregular_sampling.csv", generate_irregular_sampling),
        ("syn_spike_contamination.csv", generate_spike_contamination),
        ("syn_multi_cycle.csv", generate_multi_cycle),
    ]
    
    generated = []
    for filename, generator in generators:
        filepath = output_dir / filename
        generator(filepath)
        generated.append(str(filepath))
    
    return generated


def generate_sensor_dropout(filepath: Path) -> None:
    """30-second zero-reading gap mid-ramp.
    
    Simulates sensor dropout during heating ramp.
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 10)
    
    ambient_duration = 60
    ramp_before_dropout = 600
    dropout_duration = 30
    ramp_after_dropout = 600
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for i in range(ramp_before_dropout):
        t = ambient_temp + (ramp_rate / 60.0) * i
        temps.append(t)
    
    dropout_start_temp = temps[-1]
    for _ in range(dropout_duration):
        temps.append(0.0)
    
    resume_temp = dropout_start_temp + (ramp_rate / 60.0) * dropout_duration
    for i in range(ramp_after_dropout):
        t = resume_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * dwell_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_irregular_sampling(filepath: Path) -> None:
    """Random jitter on sample intervals.
    
    Simulates data logger with timing jitter.
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 11)
    
    n_points = 1500
    base_interval = 1.0
    jitter_std = 0.3
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    
    timestamps = []
    temps = []
    
    current_time = datetime(2025, 1, 1, 0, 0, 0)
    current_temp = ambient_temp
    
    for i in range(60):
        timestamps.append(current_time)
        temps.append(ambient_temp)
        interval = max(0.5, base_interval + np.random.normal(0, jitter_std))
        current_time = current_time + timedelta(seconds=interval)
    
    for i in range(1200):
        timestamps.append(current_time)
        elapsed = (current_time - datetime(2025, 1, 1, 0, 1, 0)).total_seconds()
        current_temp = ambient_temp + (ramp_rate / 60.0) * max(0, elapsed)
        current_temp = min(current_temp, target_temp)
        temps.append(current_temp)
        interval = max(0.5, base_interval + np.random.normal(0, jitter_std))
        current_time = current_time + timedelta(seconds=interval)
    
    for i in range(240):
        timestamps.append(current_time)
        temps.append(target_temp)
        interval = max(0.5, base_interval + np.random.normal(0, jitter_std))
        current_time = current_time + timedelta(seconds=interval)
    
    _write_csv(filepath, timestamps, temps)


def generate_spike_contamination(filepath: Path) -> None:
    """Isolated spikes at 5× noise floor.
    
    Simulates electrical interference or sensor glitches.
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 12)
    
    ambient_duration = 60
    ramp_duration = 1200
    dwell_duration = 300
    
    ambient_temp = 25.0
    target_temp = 125.0
    ramp_rate = 5.0
    noise_floor = 0.1
    spike_magnitude = noise_floor * 5.0
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for i in range(ramp_duration):
        t = ambient_temp + (ramp_rate / 60.0) * i
        t = min(t, target_temp)
        temps.append(t)
    
    temps.extend([target_temp] * dwell_duration)
    
    spike_indices = np.random.choice(
        range(100, len(temps) - 100),
        size=15,
        replace=False
    )
    for idx in spike_indices:
        direction = np.random.choice([-1, 1])
        temps[idx] += direction * spike_magnitude * (1 + np.random.random())
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


def generate_multi_cycle(filepath: Path) -> None:
    """Two complete hot-cold cycles.
    
    Full thermal cycling profile with two complete cycles.
    """
    np.random.seed(ALGORITHM_RANDOM_SEED + 13)
    
    ambient_temp = 25.0
    hot_temp = 125.0
    cold_temp = -40.0
    ramp_rate = 5.0
    
    ambient_duration = 60
    heating_duration = 1200
    hot_dwell_duration = 300
    cooling_duration = 1980
    cold_dwell_duration = 300
    
    temps = []
    
    temps.extend([ambient_temp] * ambient_duration)
    
    for cycle in range(2):
        for i in range(heating_duration):
            start_temp = cold_temp if cycle > 0 else ambient_temp
            t = start_temp + (ramp_rate / 60.0) * i
            t = min(t, hot_temp)
            temps.append(t)
        
        temps.extend([hot_temp] * hot_dwell_duration)
        
        for i in range(cooling_duration):
            t = hot_temp - (ramp_rate / 60.0) * i
            t = max(t, cold_temp)
            temps.append(t)
        
        temps.extend([cold_temp] * cold_dwell_duration)
    
    for i in range(heating_duration):
        t = cold_temp + (ramp_rate / 60.0) * i
        t = min(t, ambient_temp)
        temps.append(t)
    
    temps.extend([ambient_temp] * ambient_duration)
    
    timestamps = _generate_timestamps(len(temps))
    _write_csv(filepath, timestamps, temps)


if __name__ == "__main__":
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "tests/classification_reference_traces/synthetic"
    files = generate_all_phase2_fixtures(output_dir)
    files.extend(generate_all_phase3_fixtures(output_dir))
    print(f"Generated {len(files)} synthetic fixtures:")
    for f in files:
        print(f"  {f}")
