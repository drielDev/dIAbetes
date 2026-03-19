"""
Script de teste rápido para o sistema de monitoramento.
Executa um pipeline mínimo para verificar se o logging e exportação de métricas estão funcionando.
"""

import time
# Importando as ferramentas de monitoramento implementadas
from src.monitoring import (
    get_logger,
    PerformanceTracker,
    get_system_metrics,
    export_metrics,
)

# Logger específico para os testes
logger = get_logger("test_monitoring")


def test_basic_logging():
    """Testa funcionalidade básica de logging."""
    # Testando diferentes níveis de log (info, debug, warning)
    logger.info("Starting monitoring test")
    logger.debug("Debug message with extra data", extra={"extra_data": {"test": True}})
    logger.warning("Warning message")
    print("✓ Basic logging works")


def test_performance_tracking():
    """Testa o PerformanceTracker (medidor de performance)."""
    # Utilizando PerformanceTracker para medir tempo e recursos consumidos
    with PerformanceTracker("test_stage", logger):
        logger.info("Simulating work...")
        # Simulação de processamento com sleep e cálculos
        time.sleep(2)
        data = [i ** 2 for i in range(100000)]
    print("✓ PerformanceTracker works")


def test_system_metrics():
    """Testa coleta de métricas do sistema."""
    # Captura das métricas atuais do sistema (CPU, memória, etc)
    metrics = get_system_metrics()
    print(f"✓ System metrics collected:")
    print(f"  - CPU: {metrics['system']['cpu_percent']}%")
    print(f"  - Memory: {metrics['system']['memory_percent']}%")
    print(f"  - Process RSS: {metrics['process']['memory_rss_mb']} MB")


def test_metrics_export():
    """Testa exportação de métricas com recomendações."""
    # Exportação das métricas coletadas para arquivo JSON
    metrics_path = export_metrics(run_id="test_run")
    print(f"✓ Metrics exported to: {metrics_path}")
    print(f"  Check the file for scaling recommendations!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MONITORING SYSTEM TEST")
    print("=" * 60 + "\n")

    # Execução dos testes em sequência
    test_basic_logging()
    test_performance_tracking()
    test_system_metrics()
    test_metrics_export()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nCheck the following directories:")
    print("  - logs/       (structured JSON logs)")
    print("  - metrics/    (performance metrics + scaling recommendations)")
    print()
