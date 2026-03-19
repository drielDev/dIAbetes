"""
Módulo de monitoramento e logging para rastreamento de performance do pipeline ML.

Funcionalidades:
- Logging estruturado em JSON para todas as etapas do pipeline
- Monitoramento de recursos do sistema (CPU, memória, disco)
- Rastreamento de tempo de execução com decorators e context managers
- Exportação de métricas para decisões de configuração de auto-scaling
"""

import logging
import logging.handlers
import json
import time
import os
import functools
import platform
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager

import psutil


# Configuração de diretórios para logs e métricas
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
METRICS_DIR = BASE_DIR / "metrics"

# Cria os diretórios se não existirem
LOG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Formatador customizado para logs em JSON
class JSONFormatter(logging.Formatter):
    """Formata registros de log como JSON em uma linha para ingestão estruturada."""

    def format(self, record: logging.LogRecord) -> str:
        # Monta o dicionário com os dados do log
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Adiciona dados extras se existirem
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        # Adiciona informação de exceção se houver
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)


# Cache de loggers já criados para evitar duplicação
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Retorna um logger configurado com handlers de console e arquivo rotativo."""
    # Retorna logger do cache se já existir
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Handler para console (formato legível para humanos)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # Handler para arquivo com rotação automática (formato JSON estruturado)
    log_file = LOG_DIR / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Armazena no cache para reutilização
    _loggers[name] = logger
    return logger


# Função para capturar métricas do sistema
def get_system_metrics() -> dict:
    """Captura a utilização atual de recursos do sistema."""
    # Coleta informações de CPU, memória, disco e processo atual
    cpu_times = psutil.cpu_times_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.path.splitdrive(os.path.abspath("/"))[0] or "/")
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()

    # Retorna dicionário com todas as métricas coletadas
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system": {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_user_percent": cpu_times.user,
            "cpu_system_percent": cpu_times.system,
            "memory_total_mb": round(mem.total / (1024 ** 2), 1),
            "memory_used_mb": round(mem.used / (1024 ** 2), 1),
            "memory_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 2),
            "disk_used_gb": round(disk.used / (1024 ** 3), 2),
            "disk_percent": disk.percent,
        },
        "process": {
            "pid": proc.pid,
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "memory_rss_mb": round(proc_mem.rss / (1024 ** 2), 2),
            "memory_vms_mb": round(proc_mem.vms / (1024 ** 2), 2),
            "threads": proc.num_threads(),
        },
    }


# Context manager para rastrear performance de blocos de código
class PerformanceTracker:
    """Coleta métricas de recursos antes e depois de um bloco de código."""

    def __init__(self, stage_name: str, logger: logging.Logger | None = None):
        self.stage_name = stage_name
        self.logger = logger or get_logger("performance")
        self.start_time: float = 0
        self.end_time: float = 0
        self.start_metrics: dict = {}
        self.end_metrics: dict = {}
        self.duration_seconds: float = 0

    def __enter__(self):
        # Captura tempo e métricas no início da execução
        self.start_time = time.perf_counter()
        self.start_metrics = get_system_metrics()
        self.logger.info(
            f"[START] {self.stage_name}",
            extra={"extra_data": {"stage": self.stage_name, "event": "start",
                                  "resources": self.start_metrics}},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Captura tempo e métricas no final da execução
        self.end_time = time.perf_counter()
        self.end_metrics = get_system_metrics()
        self.duration_seconds = self.end_time - self.start_time

        # Monta resumo com todas as métricas coletadas
        summary = self._build_summary(exc_type)

        # Loga erro se houve exceção, senão loga sucesso
        if exc_type is not None:
            self.logger.error(
                f"[FAIL] {self.stage_name} after {self.duration_seconds:.2f}s",
                extra={"extra_data": summary},
                exc_info=(exc_type, exc_val, exc_tb),
            )
        else:
            self.logger.info(
                f"[DONE] {self.stage_name} in {self.duration_seconds:.2f}s",
                extra={"extra_data": summary},
            )

        # Armazena métrica no buffer para exportação posterior
        _append_metric(summary)
        return False  # não suprime exceções

    def _build_summary(self, exc_type) -> dict:
        # Calcula variação de memória durante a execução
        mem_delta = (
            self.end_metrics["process"]["memory_rss_mb"]
            - self.start_metrics["process"]["memory_rss_mb"]
        )
        return {
            "stage": self.stage_name,
            "event": "end",
            "status": "error" if exc_type else "success",
            "duration_seconds": round(self.duration_seconds, 4),
            "memory_delta_mb": round(mem_delta, 2),
            "peak_cpu_percent": self.end_metrics["system"]["cpu_percent"],
            "peak_memory_percent": self.end_metrics["system"]["memory_percent"],
            "process_memory_rss_mb": self.end_metrics["process"]["memory_rss_mb"],
            "process_threads": self.end_metrics["process"]["threads"],
            "resources_start": self.start_metrics,
            "resources_end": self.end_metrics,
        }


# Decorator para rastrear performance de funções
def track_performance(stage_name: str | None = None):
    """Decorator que envolve uma função com PerformanceTracker."""
    def decorator(func):
        name = stage_name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceTracker(name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# Buffer para armazenar métricas antes da exportação
_metrics_buffer: list[dict] = []


def _append_metric(metric: dict):
    """Adiciona métrica ao buffer."""
    _metrics_buffer.append(metric)


def export_metrics(run_id: str | None = None) -> Path:
    """Exporta métricas coletadas para arquivo JSON e retorna o caminho."""
    # Gera ID baseado no timestamp se não fornecido
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    filepath = METRICS_DIR / f"run_{run_id}.json"

    # Monta relatório completo com métricas e recomendações
    report = {
        "run_id": run_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "system_snapshot": get_system_metrics(),
        "stages": _metrics_buffer.copy(),
        "scaling_recommendations": _generate_scaling_recommendations(),
    }

    # Escreve arquivo JSON e limpa buffer
    filepath.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    get_logger("monitoring").info(
        f"Metrics exported to {filepath}",
        extra={"extra_data": {"path": str(filepath), "stages_count": len(_metrics_buffer)}},
    )
    _metrics_buffer.clear()
    return filepath


def _generate_scaling_recommendations() -> dict:
    """Analisa métricas coletadas e gera recomendações de auto-scaling."""
    if not _metrics_buffer:
        return {"message": "No stage data available for recommendations."}

    # Calcula estatísticas agregadas de todas as etapas
    total_duration = sum(s.get("duration_seconds", 0) for s in _metrics_buffer)
    peak_cpu = max((s.get("peak_cpu_percent", 0) for s in _metrics_buffer), default=0)
    peak_mem = max((s.get("peak_memory_percent", 0) for s in _metrics_buffer), default=0)
    peak_rss = max(
        (s.get("process_memory_rss_mb", 0) for s in _metrics_buffer), default=0
    )
    max_threads = max((s.get("process_threads", 0) for s in _metrics_buffer), default=1)

    # Thresholds para alertas
    CPU_HIGH = 80
    MEM_HIGH = 75
    DURATION_WARN = 300  # segundos

    recommendations = []

    # Analisa uso de CPU e gera recomendação
    if peak_cpu > CPU_HIGH:
        recommendations.append({
            "resource": "cpu",
            "severity": "high",
            "message": f"Peak CPU usage at {peak_cpu}% — consider increasing CPU limits or adding replicas.",
            "suggested_action": "scale_horizontal_or_increase_cpu_limit",
        })
    else:
        recommendations.append({
            "resource": "cpu",
            "severity": "ok",
            "message": f"CPU usage within limits ({peak_cpu}%).",
        })

    # Analisa uso de memória e gera recomendação
    if peak_mem > MEM_HIGH:
        rec_mem_mb = int(peak_rss * 1.5)
        recommendations.append({
            "resource": "memory",
            "severity": "high",
            "message": (
                f"Peak memory usage at {peak_mem}% (RSS {peak_rss} MB) "
                f"— recommend container memory limit >= {rec_mem_mb} MB."
            ),
            "suggested_action": "increase_memory_limit",
            "suggested_limit_mb": rec_mem_mb,
        })
    else:
        recommendations.append({
            "resource": "memory",
            "severity": "ok",
            "message": f"Memory usage within limits ({peak_mem}%, RSS {peak_rss} MB).",
        })

    # Verifica se o tempo total excede o limite
    if total_duration > DURATION_WARN:
        recommendations.append({
            "resource": "compute_time",
            "severity": "warning",
            "message": (
                f"Total pipeline duration {total_duration:.1f}s exceeds {DURATION_WARN}s "
                "— consider parallelism or more powerful instance."
            ),
            "suggested_action": "scale_vertical_or_parallelize",
        })

    # Identifica a etapa mais lenta (bottleneck)
    slowest = max(_metrics_buffer, key=lambda s: s.get("duration_seconds", 0))
    recommendations.append({
        "resource": "bottleneck",
        "severity": "info",
        "stage": slowest.get("stage", "unknown"),
        "duration_seconds": slowest.get("duration_seconds", 0),
        "message": f"Slowest stage: '{slowest.get('stage')}' ({slowest.get('duration_seconds', 0):.2f}s).",
    })

    return {
        "total_duration_seconds": round(total_duration, 2),
        "peak_cpu_percent": peak_cpu,
        "peak_memory_percent": peak_mem,
        "peak_rss_mb": peak_rss,
        "max_threads": max_threads,
        "items": recommendations,
    }


# Context manager para monitoramento periódico de recursos
@contextmanager
def resource_monitor(interval_seconds: float = 5.0, logger: logging.Logger | None = None):
    """Context manager que loga uso de recursos em intervalos fixos usando thread."""
    import threading

    log = logger or get_logger("resource_monitor")
    stop_event = threading.Event()

    def _monitor():
        # Loop que coleta métricas periodicamente
        while not stop_event.is_set():
            metrics = get_system_metrics()
            log.info(
                f"Resource snapshot — "
                f"CPU {metrics['system']['cpu_percent']}% | "
                f"MEM {metrics['system']['memory_percent']}% | "
                f"RSS {metrics['process']['memory_rss_mb']} MB",
                extra={"extra_data": metrics},
            )
            stop_event.wait(interval_seconds)

    # Inicia thread daemon para monitoramento em background
    t = threading.Thread(target=_monitor, daemon=True)
    t.start()
    try:
        yield
    finally:
        # Para a thread quando o context manager termina
        stop_event.set()
        t.join(timeout=2)
