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
│   ├── 01_agrupamento.py      # Script executável da Etapa 1 (DTW + AGNES + Silhueta)
│   ├── 01_agrupamento.ipynb   # Notebook equivalente da Etapa 1
│   ├── 02_matrix_profile.py   # Script executável da Etapa 2 (Matrix Profile / stumpy)
│   ├── 02_matrix_profile.ipynb # Notebook equivalente da Etapa 2
│   ├── 03_arima.py            # Script executável da Etapa 3 (Modelagem ARIMA/SARIMAX)
│   ├── 03_arima.ipynb         # Notebook equivalente da Etapa 3
│   └── src/
│       ├── __init__.py
│       ├── preparacao.py      # Funções modulares de carga, diagnóstico, limpeza e figuras
│       ├── agrupamento.py     # Funções modulares de DTW, AGNES, silhueta e figuras
│       ├── matrix_profile.py  # Funções modulares de MP, motifs/discords, Z-score e figuras
│       └── arima_modelagem.py # Funções modulares de ADF, ARIMA/SARIMAX, resíduos e previsão
├── Dados/
│   ├── bruto/                 # Base original LD2011_2014.txt (intocada)
│   └── tratado/               # Arquivos Parquet regularizados ('1h' e '1D')
├── Documentacao/
│   ├── problema_e_base_de_dados.md # Caracterização do problema e especificação da base
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

### Etapa 1 — Agrupamento: AGNES + DTW + Silhueta

```bash
python Codigo/01_agrupamento.py
```

Ou execute `Restart & Run All` no notebook `Codigo/01_agrupamento.ipynb`.

**Saídas geradas pela Etapa 1:**
- `Resultados/matriz_dtw_diario_154.npy` (cache da matriz DTW `154 × 154`)
- `Resultados/metricas_agrupamento.json`
- `Resultados/resumo_clusters_etapa1.csv`
- `Resultados/rotulos_clientes_agrupamento.csv`
- Figuras em `Documentacao/figuras/`:
  - `etapa1_dendrograma.png`
  - `etapa1_silhueta_por_k.png`
  - `etapa1_silhueta_detalhada.png`
  - `etapa1_perfis_clusters.png`

### Etapa 2 — Matrix Profile (`stumpy`)

```bash
python Codigo/02_matrix_profile.py
```

Ou execute `Restart & Run All` no notebook `Codigo/02_matrix_profile.ipynb`.

**Saídas geradas pela Etapa 2:**
- `Resultados/mp_horario_m168_zscore.npy`, `mp_horario_m24_zscore.npy`, `mp_diario_m7_zscore.npy` e `mp_horario_m168_sem_norm.npy` (caches do Matrix Profile)
- `Resultados/metricas_matrix_profile.json`
- `Resultados/resumo_motifs_discords_etapa2.csv`
- Figuras em `Documentacao/figuras/`:
  - `etapa2_mp_visao_geral_comparacao_m.png`
  - `etapa2_motifs_sobrepostos.png`
  - `etapa2_discords_contexto.png`
  - `etapa2_efeito_zscore_mp.png`

### Etapa 3 — Modelagem Preditiva (`ARIMA / SARIMAX`)

```bash
python Codigo/03_arima.py
```

Ou execute `Restart & Run All` no notebook `Codigo/03_arima.ipynb`.

**Saídas geradas pela Etapa 3:**
- `Resultados/metricas_arima.json`
- `Resultados/tabela_adf_etapa3.csv`
- `Resultados/comparacao_modelos_arima_etapa3.csv`
- `Resultados/previsoes_teste_etapa3.csv`
- Figuras em `Documentacao/figuras/`:
  - `etapa3_serie_decomposicao_stl.png`
  - `etapa3_acf_pacf_diferenciacao.png`
  - `etapa3_diagnostico_residuos.png`
  - `etapa3_previsao_vs_teste_baselines.png`

---

## Execução Completa em Sequência

```bash
conda activate datascience
python Codigo/00_preparacao.py
python Codigo/01_agrupamento.py
python Codigo/02_matrix_profile.py
python Codigo/03_arima.py
```
