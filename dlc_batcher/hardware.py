"""Hardware detection and live load sampling (no Qt, no torch).

Two kinds of numbers are kept apart:

* Capabilities: what the machine has (total VRAM, core count, RAM). Default
  settings are derived only from these.
* Load: what is in use at the moment (VRAM used by other programs, GPU and CPU
  utilization). Load is shown to the user and used as a safety limit while
  calibrating, but is never used to set defaults.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

try:
    import psutil
except Exception:  # pragma: no cover - psutil ships with DLC
    psutil = None

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: int


@dataclass
class GpuLoad:
    index: int
    used_mb: int
    total_mb: int
    util: int  # percent
    temp: int | None = None

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)


@dataclass
class Hardware:
    cpu_name: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    gpus: list[GpuInfo] = field(default_factory=list)
    os: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_gpu(self) -> bool:
        return bool(self.gpus)

    def summary(self) -> str:
        gpu = (
            ", ".join(f"{g.name} ({g.total_mb / 1024:.0f} GB)" for g in self.gpus)
            or "no NVIDIA GPU found"
        )
        return (
            f"{self.cpu_name} · {self.physical_cores} cores · "
            f"{self.ram_gb:.0f} GB RAM · {gpu}"
        )


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        ).stdout
    except Exception:
        return ""


def _nvidia_smi() -> str | None:
    return shutil.which("nvidia-smi")


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "CPU"


def _os_name() -> str:
    name, release = platform.system(), platform.release()
    if name == "Windows":
        try:
            if int(platform.version().split(".")[-1]) >= 22000:
                release = "11"
        except ValueError:
            pass
    return f"{name} {release}"


def detect_hardware() -> Hardware:
    logical = os.cpu_count() or 1
    physical = logical
    ram = 0.0
    if psutil is not None:
        physical = psutil.cpu_count(logical=False) or logical
        ram = psutil.virtual_memory().total / 2**30
    gpus: list[GpuInfo] = []
    smi = _nvidia_smi()
    if smi:
        out = _run([smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append(GpuInfo(int(parts[0]), parts[1], int(float(parts[2]))))
                except ValueError:
                    pass
    return Hardware(
        cpu_name=_cpu_name(),
        physical_cores=physical,
        logical_cores=logical,
        ram_gb=ram,
        gpus=gpus,
        os=_os_name(),
    )


def sample_gpus() -> list[GpuLoad]:
    smi = _nvidia_smi()
    if not smi:
        return []
    out = _run(
        [
            smi,
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    loads = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            temp = int(float(parts[4])) if len(parts) > 4 else None
        except ValueError:
            temp = None
        try:
            loads.append(
                GpuLoad(int(parts[0]), int(float(parts[1])), int(float(parts[2])),
                        int(float(parts[3])), temp)
            )
        except ValueError:
            pass
    return loads


def sample_system() -> dict:
    """CPU / RAM utilization in percent (0 when psutil is unavailable)."""
    if psutil is None:
        return {"cpu": 0.0, "ram": 0.0}
    return {
        "cpu": float(psutil.cpu_percent(interval=None)),
        "ram": float(psutil.virtual_memory().percent),
    }


def gpu_is_shared(load: GpuLoad | None) -> bool:
    """True when another program is clearly using the GPU right now."""
    if load is None:
        return False
    return load.util >= 20 or load.used_mb >= max(3000, 0.2 * load.total_mb)


def default_settings(hw: Hardware, device_index: int = 0) -> dict:
    """Starting settings derived from capabilities only, never from current load.

    These are moderate values that fit an idle machine of this size.
    Calibration measures the actual optimum.
    """
    if not hw.has_gpu:
        workers = max(1, min(4, hw.physical_cores // 4))
        return {"batch_size": 4, "workers": workers, "device": "cpu"}
    gpu = next((g for g in hw.gpus if g.index == device_index), hw.gpus[0])
    vram_gb = gpu.total_mb / 1024
    if vram_gb >= 20:
        batch = 64
    elif vram_gb >= 10:
        batch = 32
    elif vram_gb >= 6:
        batch = 16
    else:
        batch = 8
    # One process per GPU by default: a single worker
    # usually saturates the GPU, and extra processes then only compete for it.
    # Auto-tune tests whether more workers actually help for a given model.
    return {"batch_size": batch, "workers": 1, "device": "auto"}


def max_reasonable_workers(hw: Hardware) -> int:
    return max(1, min(16, hw.physical_cores - 1))
