# Trabalho Prático — Mineração de Dados Temporais

Pipeline completo de preparação, agrupamento temporal (`AGNES + DTW + Silhueta`), descoberta de padrões/anomalias (`Matrix Profile`) e modelagem preditiva (`ARIMA/SARIMAX`) sobre a base pública **ElectricityLoadDiagrams20112014** (UCI #321).

---

## Estrutura do Repositório

```
.
├── AGENTS.md                  # Instruções de governança e requisitos do barema
├── EXPLICACAO.md              # Registro contínuo e direto de alterações e resultados
├── GEMINI.md                  # Ponto de entrada de regras compartilhadas
├── README.md                  # Instruções de instalação e execução
├── requirements.txt           # Versões das dependências Python
├── Codigo/
│   ├── 00_preparacao.py       # Script executável da Etapa 0 (Preparação)
│   ├── 00_preparacao.ipynb    # Notebook equivalente da Etapa 0
│   └── src/
│       ├── __init__.py
│       └── preparacao.py      # Funções modulares de carga, diagnóstico, limpeza e figuras
├── Dados/
│   ├── bruto/                 # Base original LD2011_2014.txt (intocada)
│   └── tratado/               # Arquivos Parquet regularizados ('1h' e '1D')
├── Documentacao/
│   ├── relatorio.md           # Relatório técnico completo
│   └── figuras/               # Figuras PNG geradas pelo pipeline
└── Resultados/                # Métricas JSON e tabelas CSV de resumo
```

---

## Ambiente e Dependências

- **SO:** Linux (Pop!_OS)
- **Ambiente Python:** Conda `datascience` (`/home/ettore/miniconda3/envs/datascience/bin/python`)
- Dependências fixadas da etapa atual em [requirements.txt](requirements.txt).

Ative o ambiente antes de rodar scripts ou notebooks:

```bash
conda activate datascience
```

---

## Como Executar de Ponta a Ponta

Certifique-se de que o arquivo bruto `LD2011_2014.txt` esteja presente em `Dados/bruto/`.

### Etapa 0 — Preparação e Tratamento dos Dados

Via terminal (a partir da raiz do repositório):

```bash
python Codigo/00_preparacao.py
```

Ou via Jupyter / VS Code executando `Restart & Run All` no notebook `Codigo/00_preparacao.ipynb` (usando o kernel do ambiente `datascience`).

**Saídas geradas pela Etapa 0:**
- `Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet` (`35.064 × 154`, `freq='h'`)
- `Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet` (`1.461 × 154`, `freq='D'`)
- `Dados/tratado/consumo_horario_370_clientes.parquet` (`35.064 × 370`, `freq='h'`)
- `Resultados/resumo_clientes_preparacao.csv`
- `Resultados/metricas_preparacao.json`
- Figuras em `Documentacao/figuras/`:
  - `etapa0_cobertura_clientes.png`
  - `etapa0_correcao_dst_antes_depois.png`
  - `etapa0_series_tratadas_visao_geral.png`
