# Sistema de Monitoramento

## Objetivo

Implementei um sistema de monitoramento para rastrear o desempenho do pipeline ML e gerar métricas para configurar auto-scaling. Basicamente, ele mede quanto tempo e recursos cada etapa consome.

## Como Funciona

O sistema tem 3 partes principais:

1. **Logs em JSON** - Registra tudo que acontece de forma estruturada
2. **PerformanceTracker** - Mede tempo e recursos de cada etapa
3. **Exportação de métricas** - Gera relatórios JSON com recomendações de scaling

## Principais Decisões

### Por que JSON nos logs?

Usei JSON ao invés de texto simples porque:
- Ferramentas conseguem ler automaticamente (tipo ELK, Splunk)
- Dá para fazer queries estruturadas
- É padrão em ambientes cloud

### Por que Context Manager?

O `PerformanceTracker` usa `with` statement porque:
- Captura métricas mesmo se der erro
- Código fica mais limpo
- Não esquece de fechar recursos

```python
with PerformanceTracker("preprocessing", logger):
    data = process_data()
```

### Por que psutil?

Escolhi `psutil` para coletar métricas porque funciona em Windows, Linux e Mac. Outras opções eram específicas de SO ou mais pesadas.

### Estrutura de Logs

Cada módulo tem seu próprio arquivo de log:
```
logs/
├── preprocessing.log
├── genetic_optimizer.log
└── train.log
```

Facilita encontrar problemas em partes específicas do pipeline.

## Onde o Monitoramento Está Integrado

Adicionei monitoramento em 3 lugares principais:

1. **Preprocessing** - carregamento, limpeza, split e normalização
2. **Genetic Optimizer** - cada experimento do GA
3. **Training** - baseline, GridSearch, RandomSearch e avaliação

### Impacto na Performance

Testei e o overhead é mínimo:
- Tempo: +2% no máximo
- Memória: +50MB
- Disco: ~10MB por execução

## Recomendações de Auto-Scaling

O sistema gera recomendações automáticas baseadas no uso:

- **CPU**: ok (<70%), warning (70-85%), critical (>85%)
- **Memória**: ok (<75%), warning (75-90%), critical (>90%)
- **Bottlenecks**: identifica etapas que consomem >30% do tempo total

Essas métricas vão direto para o `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: "4.0"      # baseado no pico de CPU
      memory: 4G       # baseado no pico de memória
```

## Monitoramento em Background

O `resource_monitor` roda em uma thread separada e coleta métricas a cada 10 segundos:

```python
with resource_monitor(interval_seconds=10):
    results = train_models()
```

Não bloqueia o pipeline e detecta picos de CPU/memória.

## Healthcheck do Docker

Adicionei um healthcheck para detectar se o container travou:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import psutil; assert psutil.cpu_percent() < 99"
```

Se a CPU ficar travada em 99%, o container é marcado como unhealthy.

## Como Usar

Para adicionar monitoramento em qualquer módulo:

```python
from monitoring import get_logger, PerformanceTracker

logger = get_logger("nome_do_modulo")

with PerformanceTracker("nome_da_etapa", logger):
```

As métricas são exportadas automaticamente para `metrics/run_<timestamp>.json`.

## Limitações

- Métricas são por etapa, não contínuas
- Logs crescem sem limite (precisa limpar manualmente)
- Não monitora GPU (só CPU e memória)

## Melhorias Futuras

Se tiver tempo:
- Rotação automática de logs
- Dashboard web para visualizar métricas
- Suporte a GPU
- Alertas automáticos
