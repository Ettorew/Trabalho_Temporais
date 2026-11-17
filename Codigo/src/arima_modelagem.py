"""Módulo de modelagem preditiva (Etapa 3): análise de estacionariedade, ARIMA/SARIMAX, diagnóstico e previsão."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, jarque_bera, norm, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Semente padrão e hiperparâmetros explícitos da Etapa 3
SEMENTE_PADRAO: int = 42
CLIENTE_MODELAGEM: str = "MT_206"
PERIODO_SAZONAL_DIARIO: int = 7
HORIZONTE_TESTE_DIAS: int = 28          # 4 semanas completas: 2014-12-04 a 2014-12-31
NLAGS_ACF_PACF: int = 28                # 4 ciclos semanais
LAGS_LJUNG_BOX: tuple[int, ...] = (7, 14, 21, 28)
NIVEL_CONFIANCA_IC: float = 0.95        # alpha = 0.05
DATA_OUTLIER_NATAL: pd.Timestamp = pd.Timestamp("2014-12-25")

ORDEM_MODELO_PRINCIPAL: tuple[int, int, int] = (1, 1, 1)
ORDEM_SAZONAL_PRINCIPAL: tuple[int, int, int, int] = (0, 1, 2, 7)
NOME_MODELO_PRINCIPAL: str = "SARIMA(1,1,1)(0,1,2)_7"

CANDIDATOS_MODELOS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int, int]], ...] = (
    ("ARIMA(1,1,1)", (1, 1, 1), (0, 0, 0, 0)),
    ("ARIMA(2,1,2)", (2, 1, 2), (0, 0, 0, 0)),
    ("ARIMA(7,1,1)", (7, 1, 1), (0, 0, 0, 0)),
    ("SARIMA(1,1,1)(0,1,1)_7", (1, 1, 1), (0, 1, 1, 7)),
    ("SARIMA(2,1,1)(0,1,1)_7", (2, 1, 1), (0, 1, 1, 7)),
    ("SARIMA(1,1,2)(0,1,1)_7", (1, 1, 2), (0, 1, 1, 7)),
    ("SARIMA(1,1,1)(0,1,2)_7", (1, 1, 1), (0, 1, 2, 7)),
    ("SARIMA(1,1,1)(1,1,1)_7", (1, 1, 1), (1, 1, 1, 7)),
    ("SARIMA(1,1,1)(2,1,1)_7", (1, 1, 1), (2, 1, 1, 7)),
    ("SARIMA(1,1,1)(1,1,2)_7", (1, 1, 1), (1, 1, 2, 7)),
    ("SARIMA(2,1,1)(1,1,1)_7", (2, 1, 1), (1, 1, 1, 7)),
    ("SARIMA(1,1,2)(1,1,1)_7", (1, 1, 2), (1, 1, 1, 7)),
    ("SARIMA(2,1,2)(1,1,1)_7", (2, 1, 2), (1, 1, 1, 7)),
)

# Caminhos padrão relativos à raiz do repositório
CAMINHO_DIARIO_PADRAO: Path = Path("Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet")
DIRETORIO_RESULTADOS_PADRAO: Path = Path("Resultados")
DIRETORIO_FIGURAS_PADRAO: Path = Path("Documentacao/figuras")


def separar_treino_teste_cronologico(
    serie: pd.Series,
    horizonte_teste: int = HORIZONTE_TESTE_DIAS,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Separa a série cronologicamente em treino e teste sem embaralhamento."""
    if horizonte_teste <= 0 or horizonte_teste >= len(serie):
        raise ValueError(f"Horizonte de teste inválido: {horizonte_teste} para N={len(serie)}.")

    treino = serie.iloc[:-horizonte_teste].copy()
    teste = serie.iloc[-horizonte_teste:].copy()
    if serie.index.freq is not None:
        treino = treino.asfreq(serie.index.freq)
        teste = teste.asfreq(serie.index.freq)

    info_split = {
        "n_total": int(len(serie)),
        "n_treino": int(len(treino)),
        "pct_treino": round(100.0 * len(treino) / len(serie), 2),
        "inicio_treino": str(treino.index.min().date()),
        "fim_treino": str(treino.index.max().date()),
        "media_treino_kw": round(float(treino.mean()), 2),
        "desvio_treino_kw": round(float(treino.std()), 2),
        "n_teste": int(len(teste)),
        "pct_teste": round(100.0 * len(teste) / len(serie), 2),
        "inicio_teste": str(teste.index.min().date()),
        "fim_teste": str(teste.index.max().date()),
        "media_teste_kw": round(float(teste.mean()), 2),
        "desvio_teste_kw": round(float(teste.std()), 2),
    }
    return treino, teste, info_split


def _rodar_adf_unico(
    serie_trans: pd.Series,
    escopo: str,
    nome_trans: str,
    d: int,
    d_saz: int,
    regressao: str,
) -> dict[str, Any]:
    """Executa um teste ADF com seleção automática de lags por AIC."""
    stat, pval, lags_usados, nobs, criticos, _ = adfuller(
        serie_trans.dropna().to_numpy(dtype=np.float64),
        regression=regressao,
        autolag="AIC",
    )
    estacionaria = bool((pval < 0.05) and (stat < criticos["5%"]))
    return {
        "escopo": escopo,
        "serie_transformada": nome_trans,
        "d": int(d),
        "D": int(d_saz),
        "regressao": regressao,
        "estatistica_adf": round(float(stat), 4),
        "p_valor": float(pval),
        "p_valor_fmt": f"{pval:.4e}" if pval < 1e-4 else f"{pval:.4f}",
        "lags_usados": int(lags_usados),
        "n_obs": int(nobs),
        "critico_1pct": round(float(criticos["1%"]), 4),
        "critico_5pct": round(float(criticos["5%"]), 4),
        "critico_10pct": round(float(criticos["10%"]), 4),
        "estacionaria_5pct": estacionaria,
    }


def executar_bateria_adf(
    serie_treino: pd.Series,
    serie_completa: pd.Series | None = None,
    s: int = PERIODO_SAZONAL_DIARIO,
) -> pd.DataFrame:
    """Aplica o teste ADF no nível e após diferenciação regular/sazonal no treino e na série completa."""
    escopos: list[tuple[str, pd.Series]] = [("treino", serie_treino)]
    if serie_completa is not None:
        escopos.append(("serie_completa", serie_completa))

    registros: list[dict[str, Any]] = []
    for nome_escopo, ser in escopos:
        registros.append(_rodar_adf_unico(ser, nome_escopo, "Nivel (d=0, D=0)", 0, 0, "c"))
        registros.append(_rodar_adf_unico(ser, nome_escopo, "Nivel (d=0, D=0)", 0, 0, "ct"))
        registros.append(
            _rodar_adf_unico(ser.diff(1), nome_escopo, "Diferenca regular (d=1, D=0)", 1, 0, "c")
        )
        registros.append(
            _rodar_adf_unico(
                ser.diff(s), nome_escopo, f"Diferenca sazonal (d=0, D=1, s={s})", 0, 1, "c"
            )
        )
        registros.append(
            _rodar_adf_unico(
                ser.diff(1).diff(s),
                nome_escopo,
                f"Diferenca composta (d=1, D=1, s={s})",
                1,
                1,
                "c",
            )
        )
    return pd.DataFrame(registros)


def calcular_metricas_previsao(
    y_real: pd.Series | np.ndarray,
    y_prev: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calcula MAE, RMSE, MAPE (%) e sMAPE (%) entre valores reais e previstos."""
    real = np.asarray(y_real, dtype=np.float64)
    prev = np.asarray(y_prev, dtype=np.float64)
    erro = real - prev
    mae = float(np.mean(np.abs(erro)))
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    mape = float(np.mean(np.abs(erro / np.maximum(np.abs(real), 1e-9))) * 100.0)
    smape = float(
        np.mean(2.0 * np.abs(erro) / np.maximum(np.abs(real) + np.abs(prev), 1e-9)) * 100.0
    )
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "sMAPE": round(smape, 4),
    }


def _gerar_baselines(
    treino: pd.Series,
    teste: pd.Series,
    s: int = PERIODO_SAZONAL_DIARIO,
) -> dict[str, pd.Series]:
    """Constrói previsões de referência simples (Naive, Sazonal Naive e Média Móvel 7d)."""
    h = len(teste)
    naive = pd.Series(np.full(h, float(treino.iloc[-1])), index=teste.index, name="Naive")

    ult_ciclo = treino.iloc[-s:].to_numpy(dtype=np.float64)
    repeticoes = int(np.ceil(h / s))
    snaive_vals = np.tile(ult_ciclo, repeticoes)[:h]
    snaive = pd.Series(snaive_vals, index=teste.index, name="Sazonal_Naive_s7")

    ma7 = pd.Series(
        np.full(h, float(treino.iloc[-s:].mean())),
        index=teste.index,
        name="Media_Movel_7d",
    )
    return {
        "Baseline Naive (ultimo valor)": naive,
        "Baseline Sazonal Naive (s=7)": snaive,
        "Baseline Media Movel 7d": ma7,
    }


def avaliar_modelos_candidatos(
    treino: pd.Series,
    teste: pd.Series,
    candidatos: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int, int]], ...] = CANDIDATOS_MODELOS,
    lags_lb: tuple[int, ...] = LAGS_LJUNG_BOX,
    alpha_ic: float = 1.0 - NIVEL_CONFIANCA_IC,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajusta candidatos ARIMA/SARIMAX no treino, diagnostica resíduos e avalia no teste."""
    mask_sem_natal = teste.index != DATA_OUTLIER_NATAL
    linhas_comparacao: list[dict[str, Any]] = []
    detalhes: dict[str, Any] = {"modelos": {}, "baselines": {}}

    for nome, order, seasonal_order in candidatos:
        modelo = SARIMAX(
            treino,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        res_fit = modelo.fit(disp=False)
        convergiu = bool(res_fit.mle_retvals.get("converged", True)) if hasattr(res_fit, "mle_retvals") else True

        # Descarta janela inicial transiente de inicialização do filtro de Kalman
        burn_in = max(14, order[1] + seasonal_order[1] * PERIODO_SAZONAL_DIARIO + 7)
        resid_burn = res_fit.resid.iloc[burn_in:].astype(np.float64)

        # Correção de graus de liberdade (model_df = p + q + P + Q) no teste de Ljung-Box
        model_df = int(order[0] + order[2] + seasonal_order[0] + seasonal_order[2])
        lb_dict: dict[int, dict[str, Any]] = {}
        for lag in lags_lb:
            lag_int = int(lag)
            if lag_int > model_df:
                lb_row = acorr_ljungbox(resid_burn, lags=[lag_int], model_df=model_df, return_df=True).loc[lag_int]
                lb_dict[lag_int] = {
                    "estatistica": round(float(lb_row["lb_stat"]), 4),
                    "p_valor": round(float(lb_row["lb_pvalue"]), 4),
                    "gl": int(lag_int - model_df),
                }
            else:
                lb_row_raw = acorr_ljungbox(resid_burn, lags=[lag_int], model_df=0, return_df=True).loc[lag_int]
                lb_dict[lag_int] = {
                    "estatistica": round(float(lb_row_raw["lb_stat"]), 4),
                    "p_valor": None,
                    "gl": int(lag_int - model_df),
                }
        jb_stat, jb_p = jarque_bera(resid_burn)

        fc_obj = res_fit.get_forecast(steps=len(teste))
        y_prev = fc_obj.predicted_mean.astype(np.float64)
        ic_df = fc_obj.conf_int(alpha=alpha_ic)
        ic_inf = ic_df.iloc[:, 0].astype(np.float64)
        ic_sup = ic_df.iloc[:, 1].astype(np.float64)

        cobertura_28 = float(((teste >= ic_inf) & (teste <= ic_sup)).mean() * 100.0)
        cobertura_27 = float(
            ((teste.loc[mask_sem_natal] >= ic_inf.loc[mask_sem_natal]) & (teste.loc[mask_sem_natal] <= ic_sup.loc[mask_sem_natal])).mean()
            * 100.0
        )

        met_28 = calcular_metricas_previsao(teste, y_prev)
        met_27 = calcular_metricas_previsao(teste.loc[mask_sem_natal], y_prev.loc[mask_sem_natal])

        p_vals_validos = [lb_dict[int(lag)]["p_valor"] for lag in lags_lb if lb_dict[int(lag)]["p_valor"] is not None]
        ruido_branco_ok = bool(p_vals_validos) and all(pv > 0.05 for pv in p_vals_validos)

        # Tabela de coeficientes estimados
        params_info = {}
        for p_nome in res_fit.params.index:
            params_info[str(p_nome)] = {
                "estimativa": round(float(res_fit.params[p_nome]), 4),
                "erro_padrao": round(float(res_fit.bse[p_nome]), 4),
                "z_stat": round(float(res_fit.tvalues[p_nome]), 4),
                "p_valor": float(res_fit.pvalues[p_nome]),
            }

        linhas_comparacao.append(
            {
                "modelo": nome,
                "tipo": "SARIMA/ARIMA",
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "convergiu": convergiu,
                "aic": round(float(res_fit.aic), 2),
                "bic": round(float(res_fit.bic), 2),
                "hqic": round(float(res_fit.hqic), 2),
                "media_residuo": round(float(resid_burn.mean()), 4),
                "desvio_residuo": round(float(resid_burn.std(ddof=1)), 4),
                "model_df": model_df,
                "lb_p_7": lb_dict[7]["p_valor"] if lb_dict[7]["p_valor"] is not None else np.nan,
                "lb_p_14": lb_dict[14]["p_valor"] if lb_dict[14]["p_valor"] is not None else np.nan,
                "lb_p_21": lb_dict[21]["p_valor"] if lb_dict[21]["p_valor"] is not None else np.nan,
                "lb_p_28": lb_dict[28]["p_valor"] if lb_dict[28]["p_valor"] is not None else np.nan,
                "ruido_branco_ok": ruido_branco_ok,
                "jb_stat": round(float(jb_stat), 2),
                "jb_p": float(jb_p),
                "mae_28d": met_28["MAE"],
                "rmse_28d": met_28["RMSE"],
                "mape_28d": met_28["MAPE"],
                "smape_28d": met_28["sMAPE"],
                "mae_27d_sem_natal": met_27["MAE"],
                "rmse_27d_sem_natal": met_27["RMSE"],
                "mape_27d_sem_natal": met_27["MAPE"],
                "smape_27d_sem_natal": met_27["sMAPE"],
                "cobertura_ic95_28d_pct": round(cobertura_28, 2),
                "cobertura_ic95_27d_pct": round(cobertura_27, 2),
            }
        )

        detalhes["modelos"][nome] = {
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "convergiu": convergiu,
            "burn_in": int(burn_in),
            "n_residuos": int(len(resid_burn)),
            "aic": round(float(res_fit.aic), 2),
            "bic": round(float(res_fit.bic), 2),
            "hqic": round(float(res_fit.hqic), 2),
            "loglike": round(float(res_fit.llf), 2),
            "parametros": params_info,
            "media_residuo": round(float(resid_burn.mean()), 4),
            "desvio_residuo": round(float(resid_burn.std(ddof=1)), 4),
            "model_df": model_df,
            "ljung_box": lb_dict,
            "ruido_branco_ok": ruido_branco_ok,
            "jarque_bera": {
                "estatistica": round(float(jb_stat), 2),
                "p_valor": float(jb_p),
            },
            "metricas_28d": met_28,
            "metricas_27d_sem_natal": met_27,
            "cobertura_ic95_28d_pct": round(cobertura_28, 2),
            "cobertura_ic95_27d_pct": round(cobertura_27, 2),
            "resid_burn": resid_burn,
            "y_prev": y_prev,
            "ic_inf": ic_inf,
            "ic_sup": ic_sup,
        }

    # Baselines de referência sobre a mesma janela de teste
    baselines = _gerar_baselines(treino, teste, s=PERIODO_SAZONAL_DIARIO)
    for nome_b, y_b in baselines.items():
        met_28 = calcular_metricas_previsao(teste, y_b)
        met_27 = calcular_metricas_previsao(teste.loc[mask_sem_natal], y_b.loc[mask_sem_natal])
        linhas_comparacao.append(
            {
                "modelo": nome_b,
                "tipo": "Baseline",
                "order": "-",
                "seasonal_order": "-",
                "convergiu": True,
                "aic": np.nan,
                "bic": np.nan,
                "hqic": np.nan,
                "media_residuo": np.nan,
                "desvio_residuo": np.nan,
                "model_df": np.nan,
                "lb_p_7": np.nan,
                "lb_p_14": np.nan,
                "lb_p_21": np.nan,
                "lb_p_28": np.nan,
                "ruido_branco_ok": np.nan,
                "jb_stat": np.nan,
                "jb_p": np.nan,
                "mae_28d": met_28["MAE"],
                "rmse_28d": met_28["RMSE"],
                "mape_28d": met_28["MAPE"],
                "smape_28d": met_28["sMAPE"],
                "mae_27d_sem_natal": met_27["MAE"],
                "rmse_27d_sem_natal": met_27["RMSE"],
                "mape_27d_sem_natal": met_27["MAPE"],
                "smape_27d_sem_natal": met_27["sMAPE"],
                "cobertura_ic95_28d_pct": np.nan,
                "cobertura_ic95_27d_pct": np.nan,
            }
        )
        detalhes["baselines"][nome_b] = {
            "metricas_28d": met_28,
            "metricas_27d_sem_natal": met_27,
            "y_prev": y_b,
        }

    df_comparacao = pd.DataFrame(linhas_comparacao)
    return df_comparacao, detalhes


def _formatar_eixos_acf_pacf(ax: plt.Axes, nlags: int = NLAGS_ACF_PACF) -> None:
    """Adiciona marcadores verticais nos múltiplos semanais (7, 14, 21, 28) e ajusta limites."""
    for lag_saz in range(PERIODO_SAZONAL_DIARIO, nlags + 1, PERIODO_SAZONAL_DIARIO):
        ax.axvline(lag_saz, color="#d62728", linestyle="--", lw=0.85, alpha=0.55)
    ax.set_xticks(np.arange(0, nlags + 1, 7))
    ax.set_ylim(-0.65, 1.05)


def gerar_figuras_arima(
    serie_completa: pd.Series,
    treino: pd.Series,
    teste: pd.Series,
    df_adf: pd.DataFrame,
    detalhes_modelos: dict[str, Any],
    nome_modelo_principal: str = NOME_MODELO_PRINCIPAL,
    diretorio_figuras: Path | str = DIRETORIO_FIGURAS_PADRAO,
) -> list[str]:
    """Gera e salva as 4 figuras diagnósticas e preditivas da Etapa 3 em Documentacao/figuras/."""
    dir_fig = Path(diretorio_figuras)
    dir_fig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})

    arquivos_salvos: list[str] = []

    # ------------------------------------------------------------------
    # Figura 12 (etapa3_serie_decomposicao_stl.png)
    # ------------------------------------------------------------------
    stl_res = STL(serie_completa, period=PERIODO_SAZONAL_DIARIO, seasonal=15, trend=365, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 11.2))

    # Painel 1: Série original separando treino e teste + tendência STL
    axes[0].plot(treino.index, treino.values, color="#1f77b4", lw=0.9, label="Treino (2011-01-01 a 2014-12-03, n=1.433)")
    axes[0].plot(teste.index, teste.values, color="#d62728", lw=1.4, label="Teste (2014-12-04 a 2014-12-31, n=28)")
    axes[0].plot(stl_res.trend.index, stl_res.trend.values, color="#111111", lw=1.8, ls="--", label="Tendência de longo prazo (STL)")
    axes[0].axvline(teste.index.min(), color="#d62728", ls=":", lw=1.2)
    axes[0].set_title(f"Série Diária {serie_completa.name} (2011–2014): Divisão Cronológica Treino/Teste e Decomposição STL (s=7)")
    axes[0].set_ylabel("Consumo médio (kW)")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper right", ncol=3, fontsize=8.3)

    # Painel 2: Componente de tendência STL
    axes[1].plot(stl_res.trend.index, stl_res.trend.values, color="#2ca02c", lw=1.8)
    axes[1].set_title("Componente de Tendência (STL, janela anual) — Retração Plurianual 2011–2014")
    axes[1].set_ylabel("Tendência (kW)")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Painel 3: Componente sazonal semanal (zoom nos últimos 180 dias para legibilidade)
    saz_zoom = stl_res.seasonal.iloc[-180:]
    axes[2].plot(saz_zoom.index, saz_zoom.values, color="#ff7f0e", lw=1.2)
    axes[2].axhline(0.0, color="#555555", lw=0.8, ls="--")
    axes[2].set_title("Componente Sazonal Semanal (STL, period=7 dias — Zoom últimos 180 dias de 2014)")
    axes[2].set_ylabel("Sazonalidade (kW)")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))

    # Painel 4: Componente residual STL
    axes[3].plot(stl_res.resid.index, stl_res.resid.values, color="#9467bd", lw=0.8)
    axes[3].axhline(0.0, color="#555555", lw=0.8, ls="--")
    anotacoes_feriados = [
        ("2013-12-25", "Natal 2013", -155, -22, "right"),
        ("2014-01-01", "Ano Novo 2014", 45, -22, "left"),
        ("2014-12-25", "Natal 2014", -95, -22, "right"),
    ]
    for dt_feriado, rot, dx_dias, dy_kw, alinhamento in anotacoes_feriados:
        ts_f = pd.Timestamp(dt_feriado)
        if ts_f in stl_res.resid.index:
            val_f = float(stl_res.resid.loc[ts_f])
            axes[3].scatter([ts_f], [val_f], color="#d62728", s=30, zorder=5)
            axes[3].annotate(
                rot,
                xy=(ts_f, val_f),
                xytext=(ts_f + pd.Timedelta(days=dx_dias), val_f + dy_kw),
                ha=alinhamento,
                fontsize=8.0,
                color="#900000",
                arrowprops={"arrowstyle": "->", "color": "#900000", "lw": 0.8},
            )
    axes[3].set_title("Componente Residual (STL) — Choques Negativos Pontuais em Feriados")
    axes[3].set_ylabel("Resíduo (kW)")
    axes[3].set_xlabel("Data (2011–2014)")
    axes[3].set_ylim(float(stl_res.resid.min()) - 38, float(stl_res.resid.max()) + 20)
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    p12 = dir_fig / "etapa3_serie_decomposicao_stl.png"
    fig.savefig(p12, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p12))

    # ------------------------------------------------------------------
    # Figura 13 (etapa3_acf_pacf_diferenciacao.png): ACF e PACF antes/depois
    # ------------------------------------------------------------------
    adf_tr = df_adf[(df_adf["escopo"] == "treino") & (df_adf["regressao"] == "c")].set_index("serie_transformada")
    r_niv = adf_tr.loc["Nivel (d=0, D=0)"]
    r_d1 = adf_tr.loc["Diferenca regular (d=1, D=0)"]
    r_d1D1 = adf_tr.loc[f"Diferenca composta (d=1, D=1, s={PERIODO_SAZONAL_DIARIO})"]

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 11.0))

    linhas_acf = [
        (
            treino,
            f"ACF — Treino em Nível (d=0, D=0) | ADF={r_niv['estatistica_adf']:.2f} (p={r_niv['p_valor']:.4f})",
            f"PACF — Treino em Nível (d=0, D=0) | Não estacionária",
        ),
        (
            treino.diff(1).dropna(),
            f"ACF — 1ª Diferença (d=1, D=0) | ADF={r_d1['estatistica_adf']:.2f} (p={r_d1['p_valor']:.2e})",
            "PACF — 1ª Diferença (d=1, D=0) | Picos sazonais em 7, 14, 21",
        ),
        (
            treino.diff(1).diff(PERIODO_SAZONAL_DIARIO).dropna(),
            f"ACF — Dif. Composta (d=1, D=1, s=7) | ADF={r_d1D1['estatistica_adf']:.2f} (p={r_d1D1['p_valor']:.2e})",
            "PACF — Dif. Composta (d=1, D=1, s=7) | Estacionária",
        ),
    ]

    for row_idx, (ser_trans, tit_acf, tit_pacf) in enumerate(linhas_acf):
        plot_acf(ser_trans, ax=axes[row_idx, 0], lags=NLAGS_ACF_PACF, zero=False, title=tit_acf)
        plot_pacf(ser_trans, ax=axes[row_idx, 1], lags=NLAGS_ACF_PACF, zero=False, method="ywm", title=tit_pacf)
        _formatar_eixos_acf_pacf(axes[row_idx, 0], NLAGS_ACF_PACF)
        _formatar_eixos_acf_pacf(axes[row_idx, 1], NLAGS_ACF_PACF)
        axes[row_idx, 0].set_ylabel("Autocorrelação")
        axes[row_idx, 1].set_ylabel("Autocorrelação Parcial")
        if row_idx == 0:
            axes[row_idx, 0].set_ylim(-0.15, 1.05)
            axes[row_idx, 1].set_ylim(-0.35, 1.05)

    axes[2, 0].set_xlabel("Defasagem (lag em dias; linhas tracejadas = múltiplos de 7)")
    axes[2, 1].set_xlabel("Defasagem (lag em dias; linhas tracejadas = múltiplos de 7)")

    plt.tight_layout()
    p13 = dir_fig / "etapa3_acf_pacf_diferenciacao.png"
    fig.savefig(p13, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p13))

    # ------------------------------------------------------------------
    # Figura 14 (etapa3_diagnostico_residuos.png): Resíduos do modelo escolhido
    # ------------------------------------------------------------------
    info_m = detalhes_modelos["modelos"][nome_modelo_principal]
    resid = info_m["resid_burn"]
    resid_z = (resid - resid.mean()) / resid.std(ddof=1)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    # Topo esquerdo: Resíduos padronizados no tempo
    axes[0, 0].plot(resid_z.index, resid_z.values, color="#1f77b4", lw=0.75)
    axes[0, 0].axhline(0.0, color="#222222", lw=1.0, ls="--")
    axes[0, 0].axhline(3.0, color="#d62728", lw=0.9, ls=":")
    axes[0, 0].axhline(-3.0, color="#d62728", lw=0.9, ls=":")
    axes[0, 0].set_title(
        f"Resíduos Padronizados — {nome_modelo_principal} (μ={info_m['media_residuo']:.3f} kW)"
    )
    axes[0, 0].set_xlabel("Data (treino, após burn-in)")
    axes[0, 0].set_ylabel("Resíduo padronizado (z)")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Topo direito: Histograma + KDE vs Normal(0, 1)
    x_grid = np.linspace(max(-6.5, float(resid_z.min())), min(6.5, float(resid_z.max())), 300)
    axes[0, 1].hist(resid_z.values, bins=45, density=True, color="#1f77b4", alpha=0.55, edgecolor="white", label="Histograma dos resíduos")
    kde = gaussian_kde(resid_z.values)
    axes[0, 1].plot(x_grid, kde(x_grid), color="#ff7f0e", lw=2.0, label="Densidade empírica (KDE)")
    axes[0, 1].plot(x_grid, norm.pdf(x_grid, 0, 1), color="#d62728", lw=1.8, ls="--", label="Normal padrão N(0, 1)")
    axes[0, 1].set_xlim(-6.0, 6.0)
    axes[0, 1].set_title(
        f"Distribuição dos Resíduos (Jarque-Bera={info_m['jarque_bera']['estatistica']:.1f}, p={info_m['jarque_bera']['p_valor']:.1e})"
    )
    axes[0, 1].set_xlabel("Resíduo padronizado (z)")
    axes[0, 1].set_ylabel("Densidade")
    axes[0, 1].legend(loc="upper right", fontsize=8.3)

    # Baixo esquerdo: Q-Q Plot Normal
    (osm, osr), (slope, intercept, _) = probplot(resid_z.values, dist="norm")
    axes[1, 0].scatter(osm, osr, s=10, alpha=0.6, color="#1f77b4", label="Quantis observados")
    axes[1, 0].plot(osm, slope * osm + intercept, color="#d62728", lw=1.8, ls="--", label="Reta teórica Normal")
    axes[1, 0].set_title("Gráfico Q-Q Normal (Caudas Pesadas em Feriados)")
    axes[1, 0].set_xlabel("Quantis teóricos N(0, 1)")
    axes[1, 0].set_ylabel("Quantis dos resíduos padronizados")
    axes[1, 0].legend(loc="upper left", fontsize=8.5)

    # Baixo direito: Correlograma ACF dos resíduos + p-valores Ljung-Box
    plot_acf(resid, ax=axes[1, 1], lags=NLAGS_ACF_PACF, zero=False, title="ACF dos Resíduos e Teste de Ljung-Box (Ruído Branco)")
    _formatar_eixos_acf_pacf(axes[1, 1], NLAGS_ACF_PACF)
    axes[1, 1].set_ylim(-0.18, 0.18)
    axes[1, 1].set_xlabel("Defasagem (lag em dias)")
    axes[1, 1].set_ylabel("Autocorrelação dos resíduos")

    lb_linhas = [f"Ljung-Box (H0: ruído branco, model_df={info_m['model_df']}):"]
    for lag in LAGS_LJUNG_BOX:
        lb_v = info_m["ljung_box"][lag]
        p_txt = f"{lb_v['p_valor']:.4f}" if lb_v["p_valor"] is not None else "N/A"
        lb_linhas.append(f"  lag {lag:2d} (gl={lb_v['gl']:2d}): Q={lb_v['estatistica']:5.2f}, p={p_txt}")
    axes[1, 1].text(
        0.97,
        0.95,
        "\n".join(lb_linhas),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="right",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f7f9fc", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )

    plt.tight_layout()
    p14 = dir_fig / "etapa3_diagnostico_residuos.png"
    fig.savefig(p14, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p14))

    # ------------------------------------------------------------------
    # Figura 15 (etapa3_previsao_vs_teste_baselines.png)
    # ------------------------------------------------------------------
    y_prev_sarima = info_m["y_prev"]
    ic_inf = info_m["ic_inf"]
    ic_sup = info_m["ic_sup"]
    y_prev_arima = detalhes_modelos["modelos"]["ARIMA(1,1,1)"]["y_prev"]
    y_prev_snaive = detalhes_modelos["baselines"]["Baseline Sazonal Naive (s=7)"]["y_prev"]
    y_prev_naive = detalhes_modelos["baselines"]["Baseline Naive (ultimo valor)"]["y_prev"]

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 9.2))

    # Topo: Contexto (últimos 90 dias de treino) + Teste (28 dias) com IC 95%
    treino_ctx = treino.iloc[-90:]
    axes[0].plot(treino_ctx.index, treino_ctx.values, color="#1f77b4", lw=1.2, label="Treino (últimos 90 dias)")
    axes[0].plot(teste.index, teste.values, "o-", color="#111111", ms=4.0, lw=1.6, label="Teste Real (28 dias)")
    axes[0].plot(y_prev_sarima.index, y_prev_sarima.values, "s-", color="#d62728", ms=4.0, lw=1.8, label=f"{nome_modelo_principal} (Previsão)")
    axes[0].fill_between(ic_inf.index, ic_inf.values, ic_sup.values, color="#d62728", alpha=0.18, label="Intervalo de Confiança 95%")
    axes[0].axvline(teste.index.min(), color="#555555", ls="--", lw=1.2, label="Início do horizonte de teste (2014-12-04)")
    if DATA_OUTLIER_NATAL in teste.index:
        val_natal = float(teste.loc[DATA_OUTLIER_NATAL])
        axes[0].annotate(
            f"Natal 25/12\n({val_natal:.1f} kW — Discord #1)",
            xy=(DATA_OUTLIER_NATAL, val_natal),
            xytext=(DATA_OUTLIER_NATAL - pd.Timedelta(days=19), val_natal - 5),
            fontsize=8.3,
            color="#900000",
            arrowprops={"arrowstyle": "->", "color": "#900000", "lw": 0.9},
        )
    axes[0].set_title(f"Previsão Fora da Amostra — {nome_modelo_principal} com IC 95% (Contexto + Horizonte de 28 Dias)")
    axes[0].set_ylabel("Consumo médio diário (kW)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    axes[0].legend(loc="lower left", ncol=2, fontsize=8.3)

    # Baixo: Zoom exclusivo no período de teste comparando SARIMA, ARIMA(1,1,1) e baselines
    m_sar = info_m["metricas_28d"]
    m_sar_sn = info_m["metricas_27d_sem_natal"]
    m_ari = detalhes_modelos["modelos"]["ARIMA(1,1,1)"]["metricas_28d"]
    m_snv = detalhes_modelos["baselines"]["Baseline Sazonal Naive (s=7)"]["metricas_28d"]
    m_nv = detalhes_modelos["baselines"]["Baseline Naive (ultimo valor)"]["metricas_28d"]

    axes[1].plot(teste.index, teste.values, "o-", color="#111111", ms=5.0, lw=2.0, label="Real (Teste)")
    axes[1].plot(
        y_prev_sarima.index,
        y_prev_sarima.values,
        "s-",
        color="#d62728",
        ms=4.5,
        lw=1.9,
        label=f"{nome_modelo_principal} (MAE={m_sar['MAE']:.2f} | s/Natal={m_sar_sn['MAE']:.2f} kW)",
    )
    axes[1].fill_between(ic_inf.index, ic_inf.values, ic_sup.values, color="#d62728", alpha=0.14, label="IC 95% SARIMA")
    axes[1].plot(
        y_prev_snaive.index,
        y_prev_snaive.values,
        "^--",
        color="#2ca02c",
        ms=4.2,
        lw=1.4,
        label=f"Baseline Sazonal Naive s=7 (MAE={m_snv['MAE']:.2f} kW)",
    )
    axes[1].plot(
        y_prev_arima.index,
        y_prev_arima.values,
        ":",
        color="#ff7f0e",
        lw=1.7,
        label=f"ARIMA(1,1,1) s/ sazonalidade (MAE={m_ari['MAE']:.2f} kW)",
    )
    axes[1].plot(
        y_prev_naive.index,
        y_prev_naive.values,
        "-.",
        color="#7f7f7f",
        lw=1.3,
        label=f"Baseline Naive último valor (MAE={m_nv['MAE']:.2f} kW)",
    )
    axes[1].set_title("Comparação Detalhada no Conjunto de Teste (2014-12-04 a 2014-12-31): Modelos vs. Baselines")
    axes[1].set_xlabel("Data no horizonte de teste")
    axes[1].set_ylabel("Consumo médio diário (kW)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m (%a)"))
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].legend(loc="lower left", ncol=2, fontsize=8.0)

    plt.tight_layout()
    p15 = dir_fig / "etapa3_previsao_vs_teste_baselines.png"
    fig.savefig(p15, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p15))

    return arquivos_salvos


def salvar_saidas_arima(
    cliente: str,
    info_split: dict[str, Any],
    df_adf: pd.DataFrame,
    df_comparacao: pd.DataFrame,
    detalhes_modelos: dict[str, Any],
    teste: pd.Series,
    figuras: list[str],
    nome_modelo_principal: str = NOME_MODELO_PRINCIPAL,
    dir_resultados: Path | str = DIRETORIO_RESULTADOS_PADRAO,
) -> dict[str, str]:
    """Persiste tabelas CSV e métricas JSON completas da Etapa 3 em Resultados/."""
    p_res = Path(dir_resultados)
    p_res.mkdir(parents=True, exist_ok=True)

    arq_adf = p_res / "tabela_adf_etapa3.csv"
    arq_comp = p_res / "comparacao_modelos_arima_etapa3.csv"
    arq_prev = p_res / "previsoes_teste_etapa3.csv"
    arq_json = p_res / "metricas_arima.json"

    df_adf.to_csv(arq_adf, index=False)
    df_comparacao.to_csv(arq_comp, index=False)

    info_vencedor = detalhes_modelos["modelos"][nome_modelo_principal]
    df_prev = pd.DataFrame(
        {
            "timestamp": teste.index.astype(str),
            "dia_semana": teste.index.day_name(),
            "real_kw": teste.values.round(4),
            "prev_sarima_kw": info_vencedor["y_prev"].values.round(4),
            "ic95_inf_kw": info_vencedor["ic_inf"].values.round(4),
            "ic95_sup_kw": info_vencedor["ic_sup"].values.round(4),
            "prev_arima_111_kw": detalhes_modelos["modelos"]["ARIMA(1,1,1)"]["y_prev"].values.round(4),
            "prev_sazonal_naive_kw": detalhes_modelos["baselines"]["Baseline Sazonal Naive (s=7)"]["y_prev"].values.round(4),
            "prev_naive_kw": detalhes_modelos["baselines"]["Baseline Naive (ultimo valor)"]["y_prev"].values.round(4),
            "prev_media_movel_7d_kw": detalhes_modelos["baselines"]["Baseline Media Movel 7d"]["y_prev"].values.round(4),
            "erro_sarima_kw": (teste.values - info_vencedor["y_prev"].values).round(4),
        }
    )
    df_prev.to_csv(arq_prev, index=False)

    modelos_serializaveis = {
        nome: {k: v for k, v in dados.items() if k not in {"resid_burn", "y_prev", "ic_inf", "ic_sup"}}
        for nome, dados in detalhes_modelos["modelos"].items()
    }
    baselines_serializaveis = {
        nome: {k: v for k, v in dados.items() if k != "y_prev"}
        for nome, dados in detalhes_modelos["baselines"].items()
    }

    payload = {
        "semente": SEMENTE_PADRAO,
        "cliente_modelagem": cliente,
        "modelo_principal": nome_modelo_principal,
        "split_cronologico": info_split,
        "adf_testes": df_adf.to_dict(orient="records"),
        "modelos_candidatos": modelos_serializaveis,
        "baselines": baselines_serializaveis,
        "figuras_geradas": figuras,
    }
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {
        "metricas_json": str(arq_json),
        "tabela_adf_csv": str(arq_adf),
        "comparacao_modelos_csv": str(arq_comp),
        "previsoes_teste_csv": str(arq_prev),
    }
