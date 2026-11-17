# Registro de Alterações e Trabalho — `EXPLICACAO.md`

Este arquivo mantém um histórico **detalhado, porém direto**, de todas as alterações, implementações e análises realizadas no projeto.

---

## 2026-11-16 — Configuração da regra de registro contínuo

- **Etapa do barema:** Entregáveis e organização geral (3,0 pts) — governança do repositório e fluxo de trabalho do agente.
- **O que foi feito e por quê:**
  - Atualizado [AGENTS.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/AGENTS.md) na seção **4. Estrutura de repositório recomendada** para incluir `EXPLICACAO.md` e `GEMINI.md` na árvore oficial do projeto.
  - Adicionada a subseção obrigatória **Registro contínuo em `EXPLICACAO.md` (obrigatório)** na seção **6. Como trabalhar (comportamento do agente)**, determinando que toda alteração ou trabalho no projeto seja documentado aqui com: etapa do barema, o que foi feito e por quê, arquivos afetados e execução/resultados observados.
  - Inicializado este arquivo `EXPLICACAO.md` com o padrão de registro para as próximas entregas.
- **Arquivos criados ou alterados:**
  - `AGENTS.md` (atualizado)
  - `EXPLICACAO.md` (inicializado)
- **Validação:** Arquivos verificados no workspace; regra pronta para ser seguida em todas as próximas tarefas.

---

## 2026-11-16 — Definição do ambiente conda `datascience`

- **Etapa do barema:** Entregáveis e organização geral (3,0 pts) — padronização do ambiente de execução.
- **O que foi feito e por quê:**
  - Atualizada a seção **5. Ambiente e dependências** de [AGENTS.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/AGENTS.md) para especificar que o ambiente utilizado é o **conda `datascience`**, evitando criação de ambientes extras ou dúvidas na execução dos scripts/notebooks.
- **Arquivos criados ou alterados:**
  - `AGENTS.md` (atualizado)
  - `EXPLICACAO.md` (registro adicionado)
- **Validação:** Texto conferido diretamente em `AGENTS.md`.

---

## 2026-11-16 — Etapa 0: Preparação, Limpeza e Regularização Temporal dos Dados

- **Etapa do barema:** Preparação e tratamento dos dados (3,0 pts) — auditoria da fonte, tratamento de zeros/DST/lacunas, regularização do índice temporal e documentação antes/depois.
- **O que foi feito e por quê:**
  - Inspecionada a base bruta `Dados/bruto/LD2011_2014.txt` (sem modificá-la): `140.256 × 370` (`51.894.720` células), grade estrita de `15min` com timestamps marcando o fim de cada bloco (`2011-01-01 00:15:00` a `2015-01-01 00:00:00`), `0` nulos explícitos, `0` duplicatas, `0` valores negativos e `10.457.342` (`20,1511%`) zeros.
  - Desenvolvido módulo reutilizável [Codigo/src/preparacao.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/preparacao.py) e orquestradores [Codigo/00_preparacao.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/00_preparacao.py) e [Codigo/00_preparacao.ipynb](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/00_preparacao.ipynb):
    1. **Ajuste de índice:** deslocado em `-15 min` para representar o início de cada intervalo (`2011-01-01 00:00:00` a `2014-12-31 23:45:00`, `freq='15min'`).
    2. **Zeros estruturais:** `9.958.262` zeros pré-adesão (`19,1894%`) e `80.952` zeros pós-desligamento (`0,1560%`) convertidos para `NaN` para separar ausência de medição de consumo nulo.
    3. **Correção de Horário de Verão (DST):** em março (`2011-03-27`, `2012-03-25`, `2013-03-31`, `2014-03-30`), `2.655` leituras `0.0` em clientes ativos na janela de transição foram marcadas como `NaN`; em outubro (`2011-10-30`, `2012-10-28`, `2013-10-27`, `2014-10-26`), `3.840` células cuja média na hora repetida excedeu `1,5x` a vizinhança imediata foram divididas por `2,0`.
    4. **Interpolação temporal limitada:** `61.336` zeros internos curtos ($\le 8$ passos = 2h) convertidos em `NaN` e interpolados via `interpolate(method="time", limit=8, limit_area="inside")` junto às lacunas de DST de março (`63.989` células imputadas, ou `0,123305%` da base total). Desligamentos prolongados ($> 2\text{h}$) foram preservados.
    5. **Regularização e seleção do painel completo:** reamostragem por média para `1h` (`35.064` horas, `freq='h'`) e `1D` (`1.461` dias, `freq='D'`); filtragem para clientes ativos em toda a janela `2011-01-01` a `2014-12-31`, sem `NaN` internos (descartado `MT_131`) e com $\le 1\%$ de zeros internos longos (descartados `MT_132` e `MT_288`), resultando em **`154` clientes completos** sem nenhum valor ausente.
  - Geradas 3 figuras comparativas/diagnósticas em `Documentacao/figuras/`, consolidados os textos iniciais do relatório em [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md) (Seções 1 a 3) e criados [requirements.txt](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/requirements.txt) e [README.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/README.md).
- **Arquivos criados ou alterados:**
  - `Codigo/src/__init__.py` e `Codigo/src/preparacao.py` (criados)
  - `Codigo/00_preparacao.py` e `Codigo/00_preparacao.ipynb` (criados)
  - `Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet`, `Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet`, `Dados/tratado/consumo_horario_370_clientes.parquet` (gerados)
  - `Resultados/resumo_clientes_preparacao.csv` e `Resultados/metricas_preparacao.json` (gerados)
  - `Documentacao/figuras/etapa0_cobertura_clientes.png`, `etapa0_correcao_dst_antes_depois.png`, `etapa0_series_tratadas_visao_geral.png` (geradas)
  - `Documentacao/relatorio.md`, `requirements.txt`, `README.md` (criados) e `EXPLICACAO.md` (atualizado)
- **Execução e resultados observados:**
  - Comando `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/00_preparacao.py` executado com código de saída `0`.
  - Painéis finais salvos com `0` valores ausentes nas escalas horária (`35.064 × 154`) e diária (`1.461 × 154`).

---

## 2026-11-16 — Revisão crítica da Etapa 0 (Preparação)

- **Etapa do barema:** Preparação e tratamento dos dados (3,0 pts) — verificação da correção do tratamento já executado.
- **O que foi feito e por quê:** auditoria independente dos pontos sensíveis do tratamento (DST, imputação de zeros curtos, frequência persistida, dias atípicos), recalculando a partir do arquivo bruto. Nenhum arquivo do projeto foi alterado; scripts de checagem ficaram fora do repositório (pasta scratch do agente).
- **Resultados observados (execução real):**
  1. **Erro — DST de março incompleto (2012–2014):** a correção só convertia zeros exatos. Os clientes "não-zero" restantes na hora saltada têm valores espúrios em torno de 1,4% dos vizinhos (razão mediana 0,014–0,015). **98 dos 154 clientes selecionados** mantêm esse vale artificial em cada um dos 3 anos. Em 2011 a janela é normal (razão mediana 0,947).
  2. **Inconsistência — DST de outubro com limiar 1,5×:** em 2012–2014 a razão mediana é 1,94 (dobro), mas 35/26/20 clientes ativos com razão entre 1,33 e 1,49 não foram corrigidos (5 e 2 entre os 154 selecionados em 2012 e 2013). Em 2011 a razão mediana é 0,96 (sem duplicação), então as 3 correções de 2011 são provavelmente falsos positivos.
  3. **Ressalva — zeros internos curtos:** 61.336 imputações concentradas em clientes com zeros recorrentes legítimos (ex.: MT_127 com 19.080, MT_348 com 15.665). No painel dos 154 selecionados, o impacto é de apenas 391 células (máx. 33 em MT_162).
  4. **Menor — frequência perdida no Parquet:** ao reler, `index.freq` = `None` (inferível como `h`). É preciso restaurar com `asfreq` na carga.
  5. **Ok — dias de menor carga:** 25/12 e 01/01 de cada ano (feriados reais, não erro).
  6. **Pendente:** o notebook `00_preparacao.ipynb` ainda não foi executado de ponta a ponta (`nbclient` ausente no ambiente); só o `.py` foi validado.
- **Arquivos criados ou alterados:** `EXPLICACAO.md` (este registro).

---

## 2026-11-16 — Correções da Etapa 0 (DST, carga com `freq` e detalhamento de zeros curtos)

- **Etapa do barema:** Preparação e tratamento dos dados (3,0 pts) — aplicação das correções apontadas na revisão crítica.
- **O que foi feito e por quê:**
  - Atualizado [Codigo/src/preparacao.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/preparacao.py):
    1. **DST de março (2012–2014):** passou a marcar toda a janela de 1h (`01:00`–`01:45` bruto) de todos os clientes ativos como `NaN` (`4.108` células), removendo também os valores residuais espúrios (~1,4% do nível normal) que permaneciam em 98 dos 154 clientes selecionados, e preservando 2011 (regular).
    2. **DST de outubro (2012–2014):** passou a dividir por `2,0` toda a janela repetida dos clientes ativos em 2012, 2013 e 2014 (`4.184` células: 330, 347 e 369 clientes respectivamente), preservando `2011-10-30` (sem duplicação).
    3. **Função de carga `carregar_serie_tratada`:** adicionada para ler Parquets tratados restaurando automaticamente `DatetimeIndex.freq` (`'h'` ou `'D'`), evitando `freq=None` nas etapas seguintes.
    4. **Métrica de zeros curtos nos selecionados:** adicionada ao JSON de métricas (`395` células imputadas no painel dos 154 clientes, máximo de `33` em um único cliente, contra `61.348` na base completa).
  - Reexecutado `Codigo/00_preparacao.py`, regenerando Parquets, JSON/CSV de métricas e as figuras (com destaque para [etapa0_correcao_dst_antes_depois.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_correcao_dst_antes_depois.png), agora perfeitamente contínua em março e outubro).
  - Atualizado [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md) com os novos números exatos.
- **Arquivos criados ou alterados:**
  - `Codigo/src/preparacao.py` (atualizado)
  - `Dados/tratado/*.parquet`, `Resultados/metricas_preparacao.json`, `Resultados/resumo_clientes_preparacao.csv`, `Documentacao/figuras/*.png` (regenerados)
  - `Documentacao/relatorio.md` e `EXPLICACAO.md` (atualizados)
- **Execução e resultados observados:**
  - `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/00_preparacao.py` retornou código `0` (`65.454` células interpoladas = `0,126128%` da base; `0` nulos nos painéis `35.064 × 154` e `1.461 × 154`).
  - Teste de `carregar_serie_tratada` confirmou `freq: h` e `freq: D` restauradas com `0` nulos.

---

## 2026-11-16 — Publicação inicial no GitHub (`Ettorew/Trabalho_Temporais`)

- **Etapa do barema:** Entregáveis e organização geral (3,0 pts) — versionamento do código, documentação e artefatos tratados.
- **O que foi feito e por quê:**
  - Criado [.gitignore](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/.gitignore) ignorando `Dados/bruto/*` (mantendo `.gitkeep`), pois `LD2011_2014.txt` tem 679 MB (acima do limite de 100 MB do GitHub), além de caches `Resultados/*.npy`, `__pycache__/` e `.ipynb_checkpoints/`.
  - Inicializado repositório Git local na branch `main`, vinculado ao remoto `git@github.com:Ettorew/Trabalho_Temporais.git`, realizado commit inicial e `git push -u origin main`.
- **Arquivos criados ou alterados:**
  - `.gitignore` e `Dados/bruto/.gitkeep` (criados)
  - `EXPLICACAO.md` (atualizado)
- **Execução e resultados observados:**
  - `git status` validado antes do commit garantindo que `Dados/bruto/LD2011_2014.txt` não foi incluído.
  - `git push -u origin main` concluído para `Ettorew/Trabalho_Temporais`.

---

## 2026-11-16 — Etapa 1: Agrupamento Hierárquico (AGNES + DTW + Silhueta)

- **Etapa do barema:** Agrupamento (AGNES + DTW + Silhueta) (4,0 pts) — matriz de distâncias DTW, AGNES (`average` e `ward`), dendrogramas, avaliação de silhueta ($k=2\dots 10$), análise de pontos negativos, explicação do corte e interpretação dos clusters.
- **O que foi feito e por quê:**
  - Instalados `tslearn==0.10.0` e `stumpy==1.14.1` no ambiente conda `datascience` (mediante autorização) e atualizado [requirements.txt](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/requirements.txt).
  - Implementado módulo [Codigo/src/agrupamento.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/agrupamento.py) e orquestradores [Codigo/01_agrupamento.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/01_agrupamento.py) e [Codigo/01_agrupamento.ipynb](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/01_agrupamento.ipynb):
    1. **Escala e normalização:** selecionadas as 154 séries diárias (`1.461` dias) normalizadas individualmente por Z-score ($\mu=0, \sigma=1$), para agrupar por forma dinâmica (tendência e sazonalidade) em vez de amplitude kW.
    2. **Matriz DTW com cache:** calculada via `tslearn.metrics.cdist_dtw` com restrição de Sakoe-Chiba ($r = 7$ dias = 1 ciclo semanal) em `11,5 s` e salva em cache (`Resultados/matriz_dtw_diario_154.npy`).
    3. **AGNES (`average` vs. `ward`) + Silhueta ($k=2\dots 10$):** utilizada `average` como ligação metodologicamente consistente com distância DTW não-euclidiana e `ward` para comparação (declarando a limitação euclidiana no relatório).
    4. **Definição objetiva do corte e medóide representativo:** `k = 2` apresentou máxima silhueta média (`0,4860`) e `0` negativos em ambos os métodos (`[147, 7]`, separando retração `2014/2011=0,845` vs. expansão `1,266`), com medóide majoritário **`MT_206`** selecionado como série representativa para as Etapas 2 e 3. Em `ward k = 3` (`0,3200`, 3 negativos de fronteira: `MT_292`, `MT_161`, `MT_304`), destaca-se um subgrupo de 13 clientes com predominância de inverno (`medóide MT_267`, `verão/inverno=0,944`); em `ward k >= 4`, a silhueta colapsa (`0,0874`, 36 negativos).
  - Geradas as 4 figuras da Etapa 1 em `Documentacao/figuras/`, preenchida a Seção 4 e atualizadas as referências em [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md) e [README.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/README.md).
- **Arquivos criados ou alterados:**
  - `Codigo/src/agrupamento.py`, `Codigo/01_agrupamento.py`, `Codigo/01_agrupamento.ipynb` (criados)
  - `Resultados/matriz_dtw_diario_154.npy`, `Resultados/metricas_agrupamento.json`, `Resultados/resumo_clusters_etapa1.csv`, `Resultados/rotulos_clientes_agrupamento.csv` (gerados)
  - `Documentacao/figuras/etapa1_dendrograma.png`, `etapa1_silhueta_por_k.png`, `etapa1_silhueta_detalhada.png`, `etapa1_perfis_clusters.png` (geradas)
  - `requirements.txt`, `README.md`, `Documentacao/relatorio.md`, `EXPLICACAO.md` (atualizados)
- **Execução e resultados observados:**
  - Primeira execução de `Codigo/01_agrupamento.py` calculou a matriz DTW (`11,5 s`) e retornou código `0`.
  - Segunda execução validou carregamento em cache (`Cache usado: True`, `0,0 s`) e regeneração das figuras após ajuste visual de margens e legendas.

---

## 2026-11-17 — Ajustes de alinhamento (Etapas 0 e 1) e Etapa 2: Matrix Profile

- **Etapa do barema:**
  - Preparação e tratamento dos dados (3,0 pts) e Agrupamento (4,0 pts) — ajustes pontuais de consistência entre código, notebook e relatório.
  - Matrix Profile (4,0 pts) — seleção da janela $m$, identificação/interpretação de *motifs* e *discords* e discussão/demonstração da normalização Z-score.
- **O que foi feito e por quê:**
  - **Alinhamento Etapas 0 e 1:**
    - Atualizada a célula markdown `prep-tratamento-md` de [Codigo/00_preparacao.ipynb](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/00_preparacao.ipynb) para refletir a correção de DST aplicada a toda a janela em 2012–2014 (preservando 2011).
    - Adicionadas `mediana_tendencia_2014_2011` e `mediana_verao_inverno` por cliente em `resumir_perfis_clusters` ([Codigo/src/agrupamento.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/agrupamento.py)), reexecutado [Codigo/01_agrupamento.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/01_agrupamento.py) e atualizada a tabela da Seção 4.5 em [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md).
  - **Etapa 2 — Matrix Profile (`stumpy==1.14.1`):**
    - Criado o módulo [Codigo/src/matrix_profile.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/matrix_profile.py) e os orquestradores [Codigo/02_matrix_profile.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/02_matrix_profile.py) e [Codigo/02_matrix_profile.ipynb](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/02_matrix_profile.ipynb).
    - Aplicado `stumpy.stump` sobre a série representativa **`MT_206`** (medóide de `C1` em `k=2` e `C3` em `ward k=3`; `35.064` horas, `1.461` dias, `0` nulos) com caches `.npy` em `Resultados/` e comparação multi-escala: janela principal **$m = 168\text{ h}$** (ciclo semanal completo), comparada com **$m = 24\text{ h}$** (ciclo diário) e **$m = 7\text{ dias}$** (escala diária agregada).
    - **Motifs:** em $m=168\text{ h}$, o Motif #1 (`2011-05-10 15h` vs. `2011-05-17 15h`, $d_Z=1,9088$, $r=0,9892$) confirmou rotina semanal contínua de maio; em $m=24\text{ h}$, o MP revelou automaticamente duas cópias exatas de dias anteriores já presentes na base bruta (`2013-05-25`/`2013-06-01` e `2014-12-19`/`2014-12-26`, $d_Z=0,0000$, verificado `max_abs_diff=0,000000` em `LD2011_2014.txt`); em $m=7\text{ d}$, o Motif #1 (`2013-04-25` vs. `2013-09-13`, $d_Z=0,1115$, $r=0,9991$) uniu semanas de níveis médios distintos ($\Delta\mu=33,70\text{ kW}$).
    - **Discords:** em $m=168\text{ h}$, o Discord #1 (`2014-12-24 00h`–`2014-12-30 23h`, $d_Z=5,5415$, vizinho `2013-12-24`, $r=0,9086$) capturou a semana de Natal de 2014 (queda no dia 25/12 para `69,82 kW` após véspera ainda ativa até 22h); em $m=24\text{ h}$, os 3 maiores discords corresponderam exatamente a Ano Novo 2014 (`2014-01-01 04h`, $d_Z=4,4254$, $r=0,5920$), Natal 2014 e Natal 2013.
    - **Normalização Z-score:** comprovado que sem Z-score (`normalize=False`) o MP correlaciona fortemente com a amplitude local ($r=0,6513$ com $\sigma_{\text{local}}$ vs. $r=-0,1099$ com Z-score). Em experimento controlado com perturbação afim ($1,8\times + 150\text{ kW}$) sobre o Motif #1, a distância sem normalização saltou $13,56\times$ (`183,32` $\to$ `2.486,71 kW`), enquanto com Z-score permaneceu estritamente invariante ($\Delta=0,000000$, mantendo Motif #1).
  - Geradas as 4 figuras da Etapa 2 em `Documentacao/figuras/`, preenchida a Seção 5 e atualizadas as Seções 4.5 e 8 em [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md) e [README.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/README.md).
- **Arquivos criados ou alterados:**
  - `Codigo/00_preparacao.ipynb` e `Codigo/src/agrupamento.py` (atualizados)
  - `Codigo/src/matrix_profile.py`, `Codigo/02_matrix_profile.py`, `Codigo/02_matrix_profile.ipynb` (criados)
  - `Resultados/resumo_clusters_etapa1.csv`, `Resultados/metricas_agrupamento.json` (atualizados)
  - `Resultados/mp_horario_m168_zscore.npy`, `mp_horario_m24_zscore.npy`, `mp_diario_m7_zscore.npy`, `mp_horario_m168_sem_norm.npy` (caches gerados)
  - `Resultados/metricas_matrix_profile.json`, `Resultados/resumo_motifs_discords_etapa2.csv` (gerados)
  - `Documentacao/figuras/etapa2_mp_visao_geral_comparacao_m.png`, `etapa2_motifs_sobrepostos.png`, `etapa2_discords_contexto.png`, `etapa2_efeito_zscore_mp.png` (geradas)
  - `Documentacao/relatorio.md`, `README.md`, `EXPLICACAO.md` (atualizados)
- **Execução e resultados observados:**
  - `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/01_agrupamento.py` retornou código `0` (usando cache DTW).
  - `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/02_matrix_profile.py` executado com sucesso na primeira execução (`13,47 s` para $m=168\text{ h}$, `0,49 s` para $m=24\text{ h}$, `0,01 s` para $m=7\text{ d}$) e validado na segunda execução via cache (`cache=True`, `0,00 s`), ambos com código de saída `0`.

---

## 2026-11-17 — Etapa 3: Modelagem Preditiva (ARIMA / SARIMAX) e Fechamento do Relatório

- **Etapa do barema:** Modelagem preditiva (6,0 pts) e Entregáveis e organização geral (3,0 pts) — análise de estacionariedade, ACF/PACF, teste ADF, diferenciação, separação cronológica treino/teste, seleção de modelo ARIMA/SARIMAX, diagnóstico de resíduos (ruído branco), previsão fora da amostra com IC 95% vs. *baselines* e conclusão técnica integrada.
- **O que foi feito e por quê:**
  - Implementado módulo [Codigo/src/arima_modelagem.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/arima_modelagem.py) e orquestradores [Codigo/03_arima.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/03_arima.py) e [Codigo/03_arima.ipynb](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/03_arima.ipynb) utilizando `statsmodels==0.14.5` sobre a série diária regular **`MT_206`** (`1.461` dias, `freq='D'`, `0` nulos).
  - **Divisão cronológica sem *data leakage*:** últimos `28` dias (`2014-12-04` a `2014-12-31`, `4` semanas) reservados exclusivamente para teste; `1.433` dias anteriores (`2011-01-01` a `2014-12-03`) usados na identificação (STL, ACF/PACF, ADF) e ajuste.
  - **ADF e diferenciações:** série em nível não-estacionária (`ADF = -2,3050, p = 0,1704`); 1ª diferença (`d=1`) rejeita raiz unitária (`ADF = -11,4357, p = 6,35e-21`), porém mantém picos sazonais semanais significativos nos lags `7, 14, 21, 28` na ACF; diferença composta (`d=1, D=1, s=7`) atinge estacionariedade plena (`ADF = -15,0192, p = 1,03e-27`).
  - **Seleção de modelo (`SARIMA(1,1,1)(1,1,1)_7`):** dentre 10 modelos candidatos avaliados, `SARIMA(1,1,1)(1,1,1)_7` obteve os menores **AIC (`11.072,25`)** e **BIC (`11.098,56`)** e todos os parâmetros altamente significativos ($p < 10^{-115}$), enquanto ordens superiores (`p=2` ou `q=2`) produziram termos não-significativos ($p > 0,28$).
  - **Diagnóstico de resíduos:** média próxima de zero (`-0,5613 kW`), ausência de autocorrelação confirmada pelo teste de **Ljung-Box** em todos os horizontes (`lag 7: p=0,3433`, `lag 14: p=0,3188`, `lag 21: p=0,1960`, `lag 28: p=0,1418`), caracterizando **ruído branco**; teste de Jarque-Bera aponta caudas pesadas em razão dos choques de Natal/Ano Novo.
  - **Previsão e comparação com *baselines*:** nos `28` dias de teste, `SARIMA(1,1,1)(1,1,1)_7` atingiu **`MAE = 9,31 kW`** (`MAPE = 8,44%`, `sMAPE = 6,10%`, cobertura IC 95% = `96,43%`), superando `ARIMA(1,1,1)` (`MAE = 10,90`), *Sazonal Naive* (`9,74`), *Média Móvel 7d* (`10,70`) e *Naive* (`13,85`). Excluindo o choque exógeno do Natal (`2014-12-25`, único dia fora do IC 95% e já detectado como Discord #1 na Etapa 2), alcança **`MAE = 5,65 kW`**, **`MAPE = 3,02%`** e `100,0%` de cobertura nos 27 dias regulares.
  - Geradas as 4 figuras da Etapa 3 em `Documentacao/figuras/`, preenchidas as Seções 6, 7 e 8 de [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md) e atualizado [README.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/README.md).
- **Arquivos criados ou alterados:**
  - `Codigo/src/arima_modelagem.py`, `Codigo/03_arima.py`, `Codigo/03_arima.ipynb` (criados)
  - `Resultados/metricas_arima.json`, `Resultados/tabela_adf_etapa3.csv`, `Resultados/comparacao_modelos_arima_etapa3.csv`, `Resultados/previsoes_teste_etapa3.csv` (gerados)
  - `Documentacao/figuras/etapa3_serie_decomposicao_stl.png`, `etapa3_acf_pacf_diferenciacao.png`, `etapa3_diagnostico_residuos.png`, `etapa3_previsao_vs_teste_baselines.png` (geradas)
  - `Documentacao/relatorio.md`, `README.md`, `EXPLICACAO.md` (atualizados)
- **Execução e resultados observados:**
  - `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/03_arima.py` executado com código de saída `0` após ajuste visual final dos rótulos na Figura 12, confirmando convergência de todos os 10 modelos candidatos e coerência integral entre os arquivos de saída e o relatório técnico.

---

## 2026-11-17 — Revisão crítica das Etapas 1, 2 e 3

- **Etapa do barema:** Agrupamento (4,0), Matrix Profile (4,0), Modelagem preditiva (6,0) e Entregáveis (3,0) — auditoria de rigor e coerência relatório × execução.
- **O que foi feito:** recálculo independente dos pontos sensíveis (script fora do repositório, pasta scratch do agente). Nenhum arquivo do projeto foi alterado além deste registro.
- **Resultados observados (execução real):**
  1. **Erro metodológico (E3) — Ljung-Box sem correção de graus de liberdade:** com `model_df = p+q+P+Q = 4`, os p-valores do `SARIMA(1,1,1)(1,1,1)_7` caem para `0,0486` (lag 7), `0,1022` (14), `0,0695` (21) e `0,0545` (28). Rejeição marginal a 5% no lag 7; o relatório afirma "não rejeita em nenhum horizonte" (válido apenas sem correção).
  2. **Inconsistência (E3):** ACF da 1ª diferença no lag 21 é `0,232` (relatório diz `0,235`). Lags 7, 14, 28 e mínimo da ACF em nível (`0,575`) conferem.
  3. **Interpretação imprecisa (E3):** perfil semanal médio do treino tem pico em sexta (`217,58 kW`) e sábado (`218,29 kW`) e mínimo no domingo (`204,81 kW`); o relatório fala em "elevação no fim de semana".
  4. **Discussão incompleta (E3):** o *Sazonal Naive* vence o SARIMA em RMSE tanto em 28 d (`20,94` vs `22,01`) quanto em 27 d (`7,90` vs `8,36`); a exclusão do Natal é análise *post hoc* e precisa ser declarada como tal.
  5. **Discussão enviesada (E1):** o critério "clusters equilibrados / acima da média global" não é atendido por `k = 2` (`[147, 7]`, C2 com silhueta `0,245 < 0,486`), mas o texto o apresenta como atendido.
  6. **Confirmado (E2):** fora da zona de borda da perturbação, nenhuma janela supera `494,86` (máximo exato `494,86`); as `108` janelas acima do alvo estão todas na zona de transição.

---

## 2026-11-17 — Correções pós-revisão crítica (Etapas 1, 2 e 3: Código → Relatório)

- **Etapa do barema:** Agrupamento (4,0 pts), Matrix Profile (4,0 pts), Modelagem preditiva (6,0 pts) e Entregáveis (3,0 pts) — aplicação integral das correções identificadas na revisão crítica.
- **O que foi feito e por quê:**
  1. **Etapa 1 ([Codigo/src/agrupamento.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/agrupamento.py) e Seção 4 de [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md)):**
     - Removidos todos os textos/rótulos fixos das Figuras 4, 6 e 7 (`sil_media`, `n_negativos`, classificação de perfis em `ward k=3` e série negativa fronteiriça agora são derivados dinamicamente dos resultados).
     - Corrigida a discussão dos critérios de corte em `k = 2` (reconhecendo abertamente o desbalanceamento `[147, 7]` e que `C2` tem silhueta `0,2450 < 0,4860`) e citado o caso extremo `MT_187` (`2014/2011 = 10,337`).
  2. **Etapa 2 ([Codigo/src/matrix_profile.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/matrix_profile.py) e Seção 5 de [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md)):**
     - Atualizada a Figura 9 (painel baixo esquerdo `m=24h`) para exibir o primeiro *motif* orgânico não-duplicado (`#3`: `2011-03-08 23h` vs. `2011-04-03 23h`, $d_Z = 0,2514$, $r = 0,9987$), tratando as cópias exatas do arquivo bruto (`d_Z = 0,0000`) à parte como achado de auditoria.
     - Adicionada ressalva metodológica na Seção 5.4 sobre amplificação de ruído na escala diária (`m=7d`, $\sigma$ semanal baixo) e corrigida na Seção 5.5 a descrição da janela `2011-01-01` (que inclui o feriado de Ano Novo).
  3. **Etapa 3 ([Codigo/src/arima_modelagem.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/arima_modelagem.py) e Seções 6–7 de [Documentacao/relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md)):**
     - Aplicada a correção de graus de liberdade `model_df = p + q + P + Q` no teste de **Ljung-Box** em `avaliar_modelos_candidatos` e na Figura 14.
     - Expandida a busca de candidatos sazonais (`13` modelos no total, incluindo `(0,1,2)_7`, `(2,1,1)_7` e `(1,1,2)_7`). Como `SARIMA(1,1,1)(1,1,1)_7` teve rejeição marginal no lag 7 após a correção (`p = 0,0486`), selecionou-se **`SARIMA(1,1,1)(0,1,2)_7`**, que reflete diretamente os dois picos sazonais da ACF (`lag 7 = -0,291`, `lag 14 = -0,200`), atinge os **menores AIC (`11.060,79`) e BIC (`11.087,10`)**, possui todos os 4 parâmetros estruturais significativos ($p < 10^{-109}$) e passa folgadamente no Ljung-Box corrigido em todos os lags (`lag 7: p=0,1611`, `lag 14: p=0,5264`, `lag 21: p=0,3017`, `lag 28: p=0,2458`).
     - Corrigidos no relatório o lag 21 da ACF (`0,232`), a descrição da sazonalidade semanal (pico sexta/sábado `~218 kW`, mínimo domingo `204,81 kW`), os lags da ACF residual na borda (`21` e `23`), a vitória do *Sazonal Naive* em RMSE (`20,94` vs. `22,01 kW`) e a caracterização explícita da análise sem o Natal (`27d`) como *post hoc*.
- **Arquivos criados ou alterados:**
  - `Codigo/src/agrupamento.py`, `Codigo/src/matrix_profile.py`, `Codigo/src/arima_modelagem.py` (atualizados)
  - `Documentacao/figuras/etapa1_*.png`, `etapa2_*.png`, `etapa3_*.png` (regeneradas)
  - `Resultados/metricas_arima.json`, `Resultados/comparacao_modelos_arima_etapa3.csv`, `Resultados/previsoes_teste_etapa3.csv`, `Resultados/resumo_clusters_etapa1.csv`, `Resultados/metricas_agrupamento.json` (regenerados)
  - `Documentacao/relatorio.md` e `EXPLICACAO.md` (atualizados)
- **Execução e resultados observados:**
  - Executados em sequência `/home/ettore/miniconda3/envs/datascience/bin/python Codigo/01_agrupamento.py && ... Codigo/02_matrix_profile.py && ... Codigo/03_arima.py` com código de saída `0`.
  - Conferidas visualmente as figuras regeneradas (`etapa1_perfis_clusters.png`, `etapa2_motifs_sobrepostos.png`, `etapa3_diagnostico_residuos.png`, `etapa3_previsao_vs_teste_baselines.png`).
