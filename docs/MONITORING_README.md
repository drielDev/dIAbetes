# Sistema de Monitoramento - Guia Rápido

## O que é isso?

Sistema que rastreia performance do pipeline ML e gera métricas para configurar auto-scaling. Mede tempo, CPU e memória de cada etapa.

## Como usar

### 1. Adicionar monitoramento em um módulo

```python
from monitoring import get_logger, PerformanceTracker

logger = get_logger("nome_do_modulo")

with PerformanceTracker("nome_da_etapa", logger):
    # seu código aqui
    resultado = processar_dados()
```

### 2. Rodar o pipeline

```bash
python src/train.py
```

### 3. Ver os resultados

**Logs:** `logs/*.log` (formato JSON)
**Métricas:** `metrics/run_<timestamp>.json`

## Exemplo de saída

```json
{
  "run_id": "run_2026-03-19_04-30-15",
  "scaling_recommendations": {
    "cpu": {
      "status": "ok",
      "peak_percent": 47.9,
      "recommendation": "Current CPU allocation is adequate"
    },
    "memory": {
      "status": "ok", 
      "peak_percent": 64.1,
      "peak_rss_mb": 29.75
    },
    "bottlenecks": [
      {
        "stage": "train.ga_optimized_models",
        "duration_seconds": 145.2,
        "percentage_of_total": 65.3
      }
    ]
  }
}
```

## Testar o sistema

```bash
python test_monitoring.py
```

Roda um teste rápido sem executar o pipeline completo.

## Docker

O sistema funciona automaticamente no Docker:

```bash
docker-compose up --build
```

Logs e métricas são salvos em `./logs` e `./metrics` no host.

## Arquivos gerados

```
logs/
├── preprocessing.log       # Logs do preprocessamento
├── genetic_optimizer.log   # Logs do algoritmo genético
└── train.log              # Logs do treinamento

metrics/
└── run_<timestamp>.json   # Métricas consolidadas + recomendações
```

## Configurar auto-scaling

Use as métricas do arquivo JSON para configurar limites no `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "4.0"      # baseado em peak_cpu_percent
      memory: 4G       # baseado em peak_memory_mb
```

## Troubleshooting

**Logs não aparecem?**
- Verifique se as pastas `logs/` e `metrics/` existem
- Cheque permissões de escrita

**Métricas estranhas?**
- CPU/memória podem variar entre execuções
- Execute múltiplas vezes para ter média

**Arquivo muito grande?**
- Logs rotacionam automaticamente (10MB por arquivo)
- Mantenha apenas os últimos 5 backups

## Mais informações

- **Arquitetura:** `docs/MONITORING_ARCHITECTURE.md`
- **Decisões técnicas:** `docs/IMPLEMENTATION_DECISIONS.md`
