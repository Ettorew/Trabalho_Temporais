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
