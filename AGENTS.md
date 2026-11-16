# AGENTS.md — Trabalho Prático: Mineração de Dados Temporais

> Arquivo de contexto para o agente (Gemini no Antigravity). Leia inteiro antes de qualquer ação.
> Coloque na raiz do repositório. No Antigravity, também pode ser copiado para `GEMINI.md` ou `.agent/rules/`.
> Idioma de toda a comunicação, código comentado, docstrings e relatório: **português do Brasil**.

---

## 1. Contexto do projeto

Trabalho prático de disciplina (PUC Minas), em grupo de até 5 pessoas, valendo **20 pts**.
Objetivo: colocar em prática um **pipeline completo de mineração de dados temporais e modelagem preditiva** sobre uma base pública (Kaggle ou UCI Machine Learning Repository).

**Escolha da base:** priorizar séries com múltiplos componentes temporais (tendência + sazonalidade) **ou** um conjunto de múltiplas séries individuais inter-relacionadas (ex.: consumo elétrico de vários clientes, tráfego de vários servidores, vendas diárias por loja/produto). Como as Etapas 1 e 2 exigem *várias séries* (clustering) e *uma série representativa* (Matrix Profile, ARIMA), a base ideal é **multi-série com histórico longo e frequência regular**.

**Entregáveis:**
1. Código Python estruturado, comentado e **executável de ponta a ponta**.
2. Relatório técnico com a fundamentação da escolha da base e a descrição do tratamento realizado (e interpretação dos resultados).

## 2. Barema (guia de prioridade do esforço)

| Critério | Peso |
|---|---|
| Preparação e tratamento dos dados (fonte, variáveis, limpeza, regularidade temporal) | 3,0 |
| Agrupamento (AGNES + DTW + Silhueta): AGNES/Ward, dendrograma, silhueta (k), interpretação | 4,0 |
| Matrix Profile (janela m, motifs, discords) | 4,0 |
| Modelagem preditiva (ACF/PACF, ADF, diferenciação, ARIMA, resíduos, previsão) | 6,0 |
| Entregáveis e organização geral (código limpo/executável + relatório fundamentado) | 3,0 |
| **Total** | **20,0** |

Toda decisão deve ser justificada de forma que um avaliador consiga mapear o que foi feito a um item do barema.

---

## 3. Requisitos técnicos por etapa

### Preparação dos dados (3,0)
- Apresentar a **fonte** (nome, URL, licença, período coberto) e o **significado de cada variável**.
- Detectar e tratar: valores ausentes, **duplicatas** (timestamps repetidos), incoerências (valores negativos impossíveis, outliers de sensor, datas fora de ordem, fuso/DST).
- Garantir **índice temporal regular** (`DatetimeIndex` com `freq` definida): reindexar, agregar (`resample`) e/ou interpolar (`interpolate(method="time")`, ou sazonal quando fizer sentido). Documentar a escolha e **quantificar** quanto foi imputado.
- Mostrar antes/depois (contagem de nulos, duplicatas, gráfico).

### Etapa 1 — Agrupamento: AGNES + DTW + Silhueta (4,0)
- Matriz de distâncias **DTW** entre todas as séries (usar `tslearn`, `dtaidistance` ou `fastdtw`; preferir implementação eficiente e, se a base for grande, usar janela Sakoe-Chiba e/ou séries agregadas/subamostradas, justificando).
- **AGNES** (aglomerativo hierárquico) com ligação **`ward` ou `average`**. Atenção: `ward` pressupõe distância euclidiana; com matriz DTW pré-computada, `average` é o escolhimento metodologicamente seguro. Se usar `ward` com DTW, **declarar a limitação no relatório** (ou comparar ambos).
- Plotar o **dendrograma** (`scipy.cluster.hierarchy`), com linha de corte.
- Calcular **Coeficiente de Silhueta** para vários k (ex.: 2..10) com `metric="precomputed"` sobre a matriz DTW; gráfico de **silhueta detalhado** (por amostra, por cluster) para cada k relevante, e curva silhueta média × k.
- Avaliar **pontos com silhueta negativa/ruidosos** (listar quais séries e discutir).
- Texto obrigatório no relatório: **como o gráfico de silhueta define objetivamente o ponto de corte do dendrograma** (k com maior silhueta média, ausência de silhuetas negativas, clusters de largura/espessura equilibradas, silhuetas acima da média global).

### Etapa 2 — Matrix Profile (4,0)
- Selecionar **uma série representativa** (justificar: ex. medóide de um cluster da Etapa 1 ou a mais completa).
- Calcular o MP com **`stumpy`** (`stumpy.stump`) — janela **m ajustada à periodicidade** (ex.: horário com ciclo diário → m=24; semanal → 168; diário com ciclo semanal → 7). Justificar m e, se possível, comparar com outro valor.
- **Motifs:** localizar o par de subsequências de menor distância, **plotar o par sobreposto** e interpretar o padrão repetitivo.
- **Discords:** localizar o ponto de maior distância (anomalia), plotar em contexto e interpretar.
- Discutir o papel da **normalização Z-score** (o MP de `stumpy` normaliza cada subsequência) em eliminar falsos positivos causados só por **diferença de escala/offset**: o discord reflete diferença de *forma*, não de amplitude. Se possível, demonstrar com exemplo (MP com e sem normalização, ou subsequência escalada artificialmente).

### Etapa 3 — ARIMA (6,0)
Seguir **exatamente esta ordem**:
1. Plotar a série e **identificar visualmente tendência e sazonalidade** (opcional e bem-vindo: decomposição `STL`/`seasonal_decompose`).
2. Plotar **ACF e PACF** (`statsmodels.graphics.tsaplots`).
3. **Teste ADF** (`statsmodels.tsa.stattools.adfuller`): reportar estatística, p-valor, lags e valores críticos; concluir sobre estacionariedade.
4. **Diferenciar** (d, e diferença sazonal D se necessário) **até estacionariedade**, repetindo ADF a cada passo (e ACF/PACF da série diferenciada).
5. **Separar treino/teste cronologicamente** (nunca embaralhar) *antes* de ajustar. Reservar a base de teste.
6. Ajustar **ARIMA(p,d,q)** (`statsmodels.tsa.arima.model.ARIMA` ou `SARIMAX` se houver sazonalidade; `pmdarima.auto_arima` pode apoiar a busca, mas a escolha deve ser **justificada por ACF/PACF e AIC/BIC**).
7. **Diagnóstico de resíduos** (`plot_diagnostics`): média ≈ 0, ausência de autocorrelação (**Ljung-Box**), normalidade aproximada (Q-Q, histograma, Jarque-Bera). Concluir se são **ruído branco**.
8. **Previsão** no horizonte do teste com **intervalos de confiança**, plotar contra a base de teste e reportar métricas (**MAE, RMSE, MAPE/sMAPE**). Comparar com um baseline simples (ingênuo / naive sazonal) para dar contexto.

---

## 4. Estrutura de repositório recomendada

Seguir a organização usual do grupo (pastas temáticas):

```
.
├── AGENTS.md
├── EXPLICACAO.md              # registro detalhado e direto de cada alteração/trabalho realizado
├── GEMINI.md
├── README.md                  # como instalar e executar de ponta a ponta
├── requirements.txt           # versões fixadas
├── Codigo/
│   ├── 00_preparacao.ipynb    # ou .py equivalente
│   ├── 01_agrupamento.ipynb
│   ├── 02_matrix_profile.ipynb
│   ├── 03_arima.ipynb
│   └── src/                   # funções reutilizáveis (.py): carga, limpeza, dtw, plots, métricas
├── Dados/
│   ├── bruto/                 # NUNCA editar arquivos aqui
│   └── tratado/               # saídas da preparação (parquet/csv)
├── Documentacao/
│   ├── relatorio.md           # relatório técnico (fonte da verdade textual)
│   └── figuras/               # PNGs gerados pelo código
└── Resultados/                # tabelas, métricas, matriz DTW em cache (.npy)
```

Regras: notebooks e `src/` devem rodar **de ponta a ponta numa execução limpa** (`Restart & Run All`) sem intervenção manual, com `random_state`/semente fixa e caminhos relativos. Se a matriz DTW for cara, **cachear** em `Resultados/` e carregar se existir.

## 5. Ambiente e dependências

- SO: Linux (Pop!_OS). Python via **conda** (ambiente **`datascience`**); não criar ambientes novos nem instalar pacotes globalmente sem avisar. Se precisar instalar algo, adicionar a `requirements.txt` e informar o comando.
- Bibliotecas esperadas: `pandas`, `numpy`, `matplotlib`, `seaborn` (opcional), `scipy`, `scikit-learn`, `statsmodels`, `stumpy`, `tslearn` e/ou `dtaidistance`, `pmdarima` (opcional), `pyarrow` (opcional), `kagglehub`/`ucimlrepo` para download (se houver acesso).
- Preferir `.py` modulares em `Codigo/src/` chamados pelos notebooks, em vez de células gigantes.

## 6. Como trabalhar (comportamento do agente)

**Fluxo de trabalho**
1. O usuário vai **enviar material** (dataset, trechos de código, resultados, rascunhos, dúvidas, prints). Antes de agir, **identifique a qual etapa do barema aquilo pertence** e diga em uma linha o que entendeu e o que vai fazer.
2. Se faltar informação essencial (qual base, frequência, nº de séries, horizonte de previsão), **pergunte de forma objetiva** (no máximo 3 perguntas) antes de gerar código extenso. Se der para seguir com uma suposição razoável, siga e **declare a suposição**.
3. Trabalhe em **incrementos pequenos e verificáveis**: uma etapa por vez, rodando o código e conferindo a saída antes de seguir.
4. Ao analisar dados recebidos, **inspecione primeiro** (`head`, `info`, `describe`, frequência, nulos, duplicatas, período) e só depois proponha o tratamento.

**Qualidade de código**
- Funções pequenas, com *docstring* em português, type hints quando útil e comentários explicando o *porquê*, não o óbvio.
- Nada de números mágicos soltos: parâmetros (m, k, janela DTW, split, ordem ARIMA) em constantes/configuração no topo, com justificativa.
- Gráficos legíveis: título, rótulos de eixos com unidade, legenda, `plt.tight_layout()`, salvos em `Documentacao/figuras/` com nome descritivo (`etapa1_dendrograma.png`).
- Reprodutibilidade: sementes fixas, sem dependência de ordem de execução oculta, sem caminhos absolutos.

**Honestidade e rigor (muito importante)**
- **Nunca invente resultados, métricas, p-valores ou interpretações.** Todo número no relatório deve vir de uma execução real. Se não executou, diga.
- Não force o resultado "bonito": se o ADF não rejeitar, a silhueta for baixa ou os resíduos não forem ruído branco, **reporte e discuta** (e tente alternativas legítimas, como outra ordem, diferença sazonal ou outro k).
- Sinalize limitações (imputação, DTW com janela, subamostragem, normalidade dos resíduos) no relatório.
- Evitar *data leakage*: nenhuma decisão de modelagem (diferenciação, ordem) usando o conjunto de teste; transformações ajustadas só no treino.

**Documentação ↔ código**
- Código e relatório **devem permanecer alinhados**. Quando eu pedir uma mudança, **diga em que direção sincronizar** (código → relatório ou relatório → código). Se o usuário não especificar, pergunte antes de alterar os dois.
- Ao concluir cada etapa, atualizar a seção correspondente em `Documentacao/relatorio.md`.

**Registro contínuo em `EXPLICACAO.md` (obrigatório)**
- Sempre que realizar alterações ou trabalho no projeto (código, dados tratados, configurações, análises ou documentação), registrar uma entrada em `EXPLICACAO.md`.
- A explicação deve ser **detalhada, porém direta** (sem enrolação), contendo:
  1. **Contexto/Etapa do barema** relacionada ao trabalho feito.
  2. **O que foi feito e por quê** (decisões técnicas, parâmetros escolhidos, justificativas).
  3. **Arquivos criados ou alterados** e o papel de cada um.
  4. **Execução e resultados observados** (comandos rodados, métricas ou validações reais).

**Não fazer**
- Não alterar `Dados/bruto/`.
- Não apagar arquivos, não mexer em Git (commit/push/reset) sem pedido explícito.
- Não baixar bases grandes sem confirmar tamanho/licença e sem que isso seja necessário.
- Não entregar blocos gigantes de código sem tê-los executado.

## 7. Estrutura do relatório (`Documentacao/relatorio.md`)

1. **Introdução e objetivo**
2. **Base de dados**: fonte, link, licença, período, frequência original, dicionário de variáveis, **fundamentação da escolha** (por que é adequada: multi-série, tendência/sazonalidade, tamanho).
3. **Preparação e tratamento**: ausentes, duplicatas, incoerências, regularização do índice (método + justificativa + % afetado), evidências antes/depois.
4. **Etapa 1 — Agrupamento**: metodologia (DTW, ligação), dendrograma, silhueta × k, silhueta detalhada, pontos negativos, **explicação do corte via silhueta**, perfil/interpretação dos clusters.
5. **Etapa 2 — Matrix Profile**: série escolhida e por quê, escolha de m, motif (figura + interpretação), discord (figura + interpretação), **discussão da normalização Z-score**.
6. **Etapa 3 — ARIMA**: tendência/sazonalidade, ACF/PACF, ADF (tabela antes/depois da diferenciação), ordem escolhida e justificativa (AIC/BIC), diagnóstico de resíduos (Ljung-Box etc.), previsão com IC vs. teste, métricas e baseline.
7. **Conclusões e limitações**
8. **Referências** (dataset, stumpy, statsmodels, tslearn, artigos de DTW/Matrix Profile) e **divisão de tarefas do grupo**, se exigido.

Estilo: técnico, direto, em português, figuras numeradas e referenciadas no texto, sem enrolação.

## 8. Checklist final antes de dar algo como "pronto"

- [ ] Fonte e variáveis descritas; limpeza e regularidade temporal documentadas e quantificadas
- [ ] Dendrograma + silhueta (média × k e detalhada) + análise de negativos + explicação do corte
- [ ] MP com m justificado; motif plotado e interpretado; discord identificado; discussão do Z-score
- [ ] Série, ACF/PACF, ADF, diferenciação, ARIMA, resíduos (ruído branco), previsão com IC vs. teste
- [ ] Código roda do zero sem erros (`Restart & Run All`) e `README.md` explica a execução
- [ ] Relatório coerente com o código e com números reais das execuções
- [ ] Figuras salvas, legíveis e referenciadas