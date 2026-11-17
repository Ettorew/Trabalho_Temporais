"""Script principal da Etapa 3 — Modelagem Preditiva (ARIMA / SARIMAX).

Executa de ponta a ponta seguindo estritamente a ordem metodológica:
1. Carrega a série diária tratada do cliente representativo MT_206 ('1D', 1.461 dias).
2. Separa treino e teste cronologicamente (nunca embaralhando) antes de ajustar modelos:
   últimos 28 dias (4 semanas) reservados para teste.
3. Executa a bateria de testes ADF (nível, d=1, D=1 e d=1+D=1) sobre o conjunto de treino
   (e sobre a série completa apenas para registro), comprovando estacionariedade após diferenciação.
4. Ajusta 10 modelos candidatos (ARIMA e SARIMA) no treino, compara AIC/BIC, realiza diagnóstico
   de resíduos (Ljung-Box, Jarque-Bera) e gera previsões com IC de 95% contra o teste e baselines.
5. Gera as 4 figuras diagnósticas em Documentacao/figuras/ e salva tabelas/JSON em Resultados/.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from Codigo.src.arima_modelagem import (  # noqa: E402
    CANDIDATOS_MODELOS,
    CLIENTE_MODELAGEM,
    HORIZONTE_TESTE_DIAS,
    LAGS_LJUNG_BOX,
    NOME_MODELO_PRINCIPAL,
    PERIODO_SAZONAL_DIARIO,
    SEMENTE_PADRAO,
    avaliar_modelos_candidatos,
    executar_bateria_adf,
    gerar_figuras_arima,
    salvar_saidas_arima,
    separar_treino_teste_cronologico,
)
from Codigo.src.preparacao import carregar_serie_tratada  # noqa: E402


def executar_pipeline_arima() -> dict:
    """Executa todas as subetapas da Etapa 3 e retorna o resumo dos resultados."""
    np.random.seed(SEMENTE_PADRAO)
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"[1/5] Carregando série diária tratada de {CLIENTE_MODELAGEM}...")
    df_diario = carregar_serie_tratada(
        RAIZ_PROJETO / "Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet",
        freq="D",
    )
    serie = df_diario[CLIENTE_MODELAGEM].copy()
    print(
        f"      Série: {len(serie)} dias (freq={serie.index.freqstr}) | "
        f"Período: {serie.index.min().date()} a {serie.index.max().date()} | "
        f"Nulos: {int(serie.isna().sum())}"
    )

    print(f"[2/5] Separando treino/teste cronologicamente (teste = últimos {HORIZONTE_TESTE_DIAS} dias)...")
    treino, teste, info_split = separar_treino_teste_cronologico(
        serie, horizonte_teste=HORIZONTE_TESTE_DIAS
    )
    print(
        f"      Treino: {info_split['n_treino']} dias ({info_split['inicio_treino']} a {info_split['fim_treino']}) | "
        f"Teste: {info_split['n_teste']} dias ({info_split['inicio_teste']} a {info_split['fim_teste']})"
    )

    print("[3/5] Executando testes ADF (nível e após diferenciações d=1, D=1 s=7)...")
    df_adf = executar_bateria_adf(
        serie_treino=treino,
        serie_completa=serie,
        s=PERIODO_SAZONAL_DIARIO,
    )
    adf_treino_c = df_adf[(df_adf["escopo"] == "treino") & (df_adf["regressao"] == "c")]
    for _, row in adf_treino_c.iterrows():
        print(
            f"      {row['serie_transformada']:34s} | ADF={row['estatistica_adf']:8.4f} | "
            f"p={row['p_valor_fmt']:>10s} | lags={row['lags_usados']:2d} | estac={row['estacionaria_5pct']}"
        )

    print(f"[4/5] Ajustando {len(CANDIDATOS_MODELOS)} modelos ARIMA/SARIMA no treino e avaliando no teste...")
    df_comparacao, detalhes_modelos = avaliar_modelos_candidatos(
        treino=treino,
        teste=teste,
        candidatos=CANDIDATOS_MODELOS,
        lags_lb=LAGS_LJUNG_BOX,
    )
    for _, row in df_comparacao.iterrows():
        if row["tipo"] == "SARIMA/ARIMA":
            print(
                f"      {row['modelo']:24s} | AIC={row['aic']:8.2f} | BIC={row['bic']:8.2f} | "
                f"LB_p14={row['lb_p_14']:.4f} LB_p28={row['lb_p_28']:.4f} | "
                f"MAE28={row['mae_28d']:5.2f} MAE27={row['mae_27d_sem_natal']:5.2f}"
            )
        else:
            print(
                f"      {row['modelo']:24s} | (Baseline)                             | "
                f"MAE28={row['mae_28d']:5.2f} MAE27={row['mae_27d_sem_natal']:5.2f}"
            )

    print("[5/5] Gerando figuras diagnósticas e persistindo saídas em Resultados/...")
    figuras = gerar_figuras_arima(
        serie_completa=serie,
        treino=treino,
        teste=teste,
        df_adf=df_adf,
        detalhes_modelos=detalhes_modelos,
        nome_modelo_principal=NOME_MODELO_PRINCIPAL,
        diretorio_figuras=RAIZ_PROJETO / "Documentacao/figuras",
    )
    figuras_rel = [str(Path(f).relative_to(RAIZ_PROJETO)) for f in figuras]

    saidas = salvar_saidas_arima(
        cliente=CLIENTE_MODELAGEM,
        info_split=info_split,
        df_adf=df_adf,
        df_comparacao=df_comparacao,
        detalhes_modelos=detalhes_modelos,
        teste=teste,
        figuras=figuras_rel,
        nome_modelo_principal=NOME_MODELO_PRINCIPAL,
        dir_resultados=RAIZ_PROJETO / "Resultados",
    )
    print("      Arquivos persistidos:")
    for chave, valor in saidas.items():
        print(f"        - {chave}: {Path(valor).relative_to(RAIZ_PROJETO)}")

    return {
        "split": info_split,
        "figuras": figuras_rel,
        "saidas": saidas,
    }


if __name__ == "__main__":
    executar_pipeline_arima()
    print("\nEtapa 3 concluída com sucesso.")
