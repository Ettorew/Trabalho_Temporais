# Caracterização do Problema e Especificação da Base de Dados

Este documento descreve de forma dedicada o **problema de mineração de dados temporais** abordado no trabalho e a **especificação técnica completa da base de dados utilizada**, consolidando metadados oficiais, dicionário de variáveis, peculiaridades diagnosticadas nos registros brutos, critérios de filtragem do painel e catálogo dos artefatos gerados.

---

## 1. Contextualização e Formulação do Problema

### 1.1 Contexto de domínio
Redes elétricas modernas monitoram o consumo de clientes de média e baixa tensão por meio de medidores inteligentes (*smart meters*) em alta frequência (blocos de 15 minutos). A base analisada registra o consumo elétrico de **370 consumidores portugueses** ao longo de **quatro anos completos (2011–2014)**, período marcado por forte heterogeneidade de perfis de uso (estabelecimentos comerciais, serviços contínuos, escritórios, instalações industriais e residenciais) e por mudanças macroeconômicas em Portugal que induziram retração ou eficiência energética na maioria dos pontos, em contraste com nichos específicos de expansão.

Fisicamente, séries de carga elétrica combinam **quatro componentes temporais sobrepostos**:
1. **Ciclo intradiário (`24 h`):** variação entre madrugada de carga base e pico comercial/operacional diurno.
2. **Ciclo semanal (`7 dias`):** alternância entre dias úteis, sexta-feira/sábado e domingo.
3. **Sazonalidade anual (`~365 dias`):** variação climática entre regimes dominados por climatização de verão e regimes dominados por aquecimento de inverno.
4. **Tendência plurianual e eventos exógenos:** trajetórias de crescimento ou retração entre 2011 e 2014, além de choques pontuais em feriados nacionais (Natal e Ano Novo) e artefatos de registro de horário de verão (*Daylight Saving Time* — DST).

### 1.2 Desafios analíticos do pipeline
O problema prático de mineração e modelagem temporal desdobra-se em quatro etapas integradas:

1. **Saneamento e regularização temporal (Etapa 0):** distinguir ausência estrutural de medição (clientes conectados após janeiro de 2011) de consumo nulo legítimo; identificar e corrigir distorções de horário de verão em março (hora suprimida) e outubro (hora repetida acumulada); imputar microfalhas curtas sem distorcer desligamentos prolongados; e regularizar a grade temporal (`15min`, `h` e `D`).
2. **Segmentação de perfis dinâmicos (Etapa 1 — DTW + AGNES + Silhueta):** agrupar múltiplos consumidores pela **forma** da trajetória temporal — invariante à potência instalada em kW e tolerante a pequenos deslocamentos temporais (via *Dynamic Time Warping* com banda de Sakoe-Chiba de 7 dias) — determinando objetivamente o número de grupos *k* pelo Coeficiente de Silhueta e elegendo um medóide representativo.
3. **Descoberta de padrões recorrentes e anomalias de forma (Etapa 2 — Matrix Profile):** varrer a série representativa (`MT_206`) via `stumpy.stump` em múltiplas escalas ($m = 168\text{ h}$, $24\text{ h}$ e $7\text{ dias}$) para extrair rotinas semanais repetitivas (*motifs*), eventos extremos e feriados (*discords*) e demonstrar a invariância à escala e *offset* proporcionada pela normalização Z-score.
4. **Modelagem preditiva e diagnóstico probabilístico (Etapa 3 — ARIMA/SARIMAX):** caracterizar estacionariedade via decomposição STL, ACF/PACF e testes ADF; ajustar modelo parcimonioso com diferenciação regular e sazonal semanal (`d=1, D=1, s=7`) estritamente sobre o histórico de treino; validar ausência de autocorrelação residual (ruído branco via Ljung-Box); e avaliar previsão multi-passo ($H = 28$ dias) com intervalos de confiança de 95% frente a *baselines* de referência.

---

## 2. Identificação Oficial e Metadados da Base de Dados

| Atributo | Especificação |
|---|---|
| **Nome oficial** | *ElectricityLoadDiagrams20112014* |
| **Repositório / ID** | UCI Machine Learning Repository — Dataset ID `321` |
| **Autor / Doador** | Artur Trindade (doado em 2015) |
| **URL oficial** | <https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014> |
| **DOI** | `10.24432/C58C86` |
| **Licença** | Creative Commons Attribution 4.0 International (**CC BY 4.0**) |
| **Arquivo bruto local** | `Dados/bruto/LD2011_2014.txt` (`679 MB`, texto delimitado por `;`, separador decimal `,`, mantido intocado) |
| **Dimensões brutas** | `140.256` linhas temporais × `370` séries (`MT_001` a `MT_370`) = `51.894.720` células |
| **Período coberto (bruto)** | `2011-01-01 00:15:00` a `2015-01-01 00:00:00` (timestamp indica o **fim** do intervalo de 15 min) |
| **Período alinhado ao início do bloco** | `2011-01-01 00:00:00` a `2014-12-31 23:45:00` (`4` anos completos = `1.461` dias, incluindo `2012` bissexto) |
| **Frequência original** | `15min` (`96` medições por dia) |
| **Frequências derivadas no pipeline** | Horária (`freq='h'`, `35.064` registros) e Diária (`freq='D'`, `1.461` registros) |

---

## 3. Dicionário de Variáveis e Significado Físico

| Variável | Tipo | Unidade | Descrição |
|---|---|---|---|
| `timestamp` (índice) | `DatetimeIndex` | ISO-8601 (`YYYY-MM-DD HH:MM:SS`) | Início do intervalo de medição após subtração de `15 min` do timestamp original de fim de bloco (fuso horário local de Portugal continental, sem marcador explícito de UTC/DST no arquivo bruto). |
| `MT_001` … `MT_370` | Numérica contínua (`float64`) | **kW** | Potência elétrica ativa média demandada pelo medidor do cliente *i* durante o intervalo de 15 minutos. |

### Convenção física de unidades na reamostragem
- Na grade original de **15 minutos** ($0,25\text{ h}$), a energia consumida no bloco é dada por $\text{kWh} = \text{kW} / 4$.
- Ao agregar por média para a grade **horária (`resample('1h').mean()`)**, a unidade permanece **kW** (potência média na hora), sendo numericamente idêntica à energia horária consumida (**kWh/h**).
- Na grade **diária (`resample('1D').mean()`)**, preserva-se a potência média diária em **kW** (equivalente a $1/24$ da energia total diária em kWh).

---

## 4. Diagnóstico Estrutural e Peculiaridades Reais da Base Bruta

A auditoria direta sobre `Dados/bruto/LD2011_2014.txt` (implementada em [preparacao.py](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Codigo/src/preparacao.py) e registrada em [metricas_preparacao.json](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Resultados/metricas_preparacao.json)) revelou:

1. **Integridade formal da grade temporal:**
   - **0** `NaN` explícitos no texto bruto;
   - **0** timestamps duplicados;
   - **0** passos irregulares ($\neq 15\text{ min}$) após o deslocamento de `-15 min`, com índice estritamente monotônico crescente (`freq='15min'`);
   - **0** valores negativos.
2. **Decomposição dos `10.457.342` zeros (`20,1511%` das 51.894.720 células):**
   - **Zeros pré-adesão (`9.958.262` = `19,1894%`):** ocorrem antes do primeiro consumo $> 0$ de clientes que só passaram a ser monitorados após janeiro de 2011 (com grandes saltos de entrada em 2012 e 2014, conforme a **Figura 1** em [etapa0_cobertura_clientes.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_cobertura_clientes.png)). Convertidos para `NaN` para não distorcer médias nem distâncias.
   - **Zeros pós-desligamento (`80.952` = `0,1560%`):** ocorrem após o último registro $> 0$ de clientes encerrados antes do fim de 2014; também convertidos para `NaN`.
   - **Zeros internos (`418.128` = `0,8057%`):** subdivididos em microfalhas curtas ($\le 8$ passos consecutivos = $\le 2\text{ h}$, com **`61.348` células** convertidas em `NaN` na base completa e apenas **`395` células** no painel final de 154 clientes) e desligamentos prolongados ($> 2\text{ h}$, preservados como `0,0` legítimo).
3. **Anomalias sistemáticas de Horário de Verão (DST em Portugal, 2012–2014):**
   - **Março (último domingo, hora suprimida — timestamps brutos `01:00` a `01:45`):** em `2012-03-25`, `2013-03-31` e `2014-03-30`, os medidores apresentaram zeros artificiais ou resíduos espúrios (~`1,4%` do nível adjacente), derrubando a carga agregada média de ~`130–160 MW` para `4.287,09 kW` (2012), `4.369,79 kW` (2013) e `454,48 kW` (2014). Toda a janela em clientes ativos (`4.108` células) foi marcada como `NaN` e interpolada; em `2011-03-27` (`69.145,14 kW`), o registro foi regular e preservado.
   - **Outubro (último domingo, hora repetida — timestamps brutos `01:00` a `01:45`):** sem desambiguação de fuso no índice, os anos de `2012-10-28`, `2013-10-27` e `2014-10-26` acumularam o consumo de **duas horas em um único bloco**, elevando a carga média agregada para `239.581,98 kW`, `248.856,39 kW` e `267.445,09 kW` (~dobro das horas adjacentes). Aplicou-se divisão por `2,0` sobre toda a janela dos clientes ativos (`330`, `347` e `369` clientes = `4.184` células), preservando `2011-10-30` (`67.374,26 kW`, regular). O efeito antes/depois consta em [etapa0_correcao_dst_antes_depois.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_correcao_dst_antes_depois.png).
4. **Interpolação total e achado adicional de auditoria (réplicas de 24h no bruto):**
   - A interpolação temporal (`method="time", limit=8, limit_area="inside"`) preencheu **`65.454` células** (`0,126128%` da base total).
   - Na Etapa 2 (*Matrix Profile* com $m=24\text{ h}$), identificou-se ainda que o arquivo bruto original `LD2011_2014.txt` contém dias completos replicados exatamente da semana anterior pelo registrador na série `MT_206` (`2013-05-25` vs. `2013-06-01` e `2014-12-19` vs. `2014-12-26`, com `max_abs_diff = 0,000000 kW` em todos os 96 blocos de 15 minutos).

---

## 5. Critérios de Seleção do Painel Completo (`154` Clientes) e Série Representativa (`MT_206`)

Para garantir comparabilidade temporal homogênea nos 4 anos (2011–2014) sem imputação artificial de anos inteiros pré-adesão:

1. **Cobertura integral e ausência de lacunas longas internas:** exigiu-se `inicio_ativo <= 2011-01-01 01:00:00`, `fim_ativo >= 2014-12-31 23:00:00` e `nans_internos_finais == 0`, atendido por **`156` clientes** (`MT_131` ficou fora por apresentar `1` `NaN` interno não interpolado e `46,89%` de zeros longos).
2. **Limite máximo de zeros internos prolongados ($\le 1,0\%$):** excluíram-se mais `2` clientes intermitentes (`MT_132` com `56,13%` e `MT_288` com `1,51%` de zeros longos).
3. **Painel consolidado:** **`154` clientes completos** (`41,6%` dos 370 originais), com **0 valores ausentes** tanto na escala horária (`35.064 × 154`) quanto na diária (`1.461 × 154`).
4. **Série representativa selecionada (`MT_206`):** eleita objetivamente ao fim da Etapa 1 por ser simultaneamente o **medóide do cluster majoritário `C1`** em `k = 2` (`147` clientes = `95,45%`, silhueta `0,5667`) e do **núcleo de verão `C3`** em `ward k = 3` (`134` clientes = `87,01%`, silhueta `0,4548`). Apresenta `100,0%` de cobertura, apenas `4` microfalhas curtas imputadas em 15 min, `0` zeros longos, média de `209,22 kW` (desvio `70,17 kW` horário; `27,53 kW` diário no treino).

---

## 6. Catálogo de Dados Tratados e Resumos Gerados

| Arquivo | Localização | Dimensões / Conteúdo | Papel no Pipeline |
|---|---|---|---|
| Base bruta original | [LD2011_2014.txt](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Dados/bruto/LD2011_2014.txt) | `140.256 × 370` (`15min`) | Fonte imutável da UCI (#321) |
| Painel horário completo | [consumo_horario_clientes_completos_2011_2014.parquet](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet) | `35.064 × 154` (`freq='h'`, `0` nulos) | Base das Etapas 1 (perfil intradiário) e 2 (`m=168h`, `24h`) |
| Painel diário completo | [consumo_diario_clientes_completos_2011_2014.parquet](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet) | `1.461 × 154` (`freq='D'`, `0` nulos) | Base das Etapas 1 (DTW/AGNES), 2 (`m=7d`) e 3 (ARIMA/SARIMAX) |
| Painel horário integral | [consumo_horario_370_clientes.parquet](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Dados/tratado/consumo_horario_370_clientes.parquet) | `35.064 × 370` (`freq='h'`, `NaN` fora da janela ativa) | Registro tratado de todos os 370 medidores |
| Auditoria por cliente | [resumo_clientes_preparacao.csv](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Resultados/resumo_clientes_preparacao.csv) | `370` linhas × `13` colunas | Cobertura, zeros imputados/preservados e estatísticas por cliente |
| Métricas da preparação | [metricas_preparacao.json](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Resultados/metricas_preparacao.json) | JSON estruturado | Todos os contadores antes/depois da Etapa 0 |

Para a visão gráfica completa da cobertura temporal, da correção de DST e dos padrões gerais das séries tratadas, consultar [etapa0_cobertura_clientes.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_cobertura_clientes.png), [etapa0_correcao_dst_antes_depois.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_correcao_dst_antes_depois.png) e [etapa0_series_tratadas_visao_geral.png](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/figuras/etapa0_series_tratadas_visao_geral.png), bem como a análise integrada em [relatorio.md](file:///home/ettore/Documentos/PUC/Temporais/Trabalho/Documentacao/relatorio.md).
