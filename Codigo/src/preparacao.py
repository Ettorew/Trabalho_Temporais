"""Módulo de preparação, limpeza e regularização temporal da base ElectricityLoadDiagrams20112014."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Semente padrão para reprodutibilidade no projeto
SEMENTE_PADRAO: int = 42

# Caminhos relativos padrão a partir da raiz do repositório
CAMINHO_BRUTO_PADRAO: Path = Path("Dados/bruto/LD2011_2014.txt")
DIRETORIO_TRATADO_PADRAO: Path = Path("Dados/tratado")
DIRETORIO_RESULTADOS_PADRAO: Path = Path("Resultados")
DIRETORIO_FIGURAS_PADRAO: Path = Path("Documentacao/figuras")

# Datas de transição do horário de verão em Portugal (último domingo de março e outubro)
DATAS_DST_MARCO: tuple[str, ...] = (
    "2011-03-27",
    "2012-03-25",
    "2013-03-31",
    "2014-03-30",
)
DATAS_DST_OUTUBRO: tuple[str, ...] = (
    "2011-10-30",
    "2012-10-28",
    "2013-10-27",
    "2014-10-26",
)
# Anos em que a base apresenta efetivamente distorção de DST (2011 manteve registro regular)
DATAS_DST_MARCO_AFETADAS: tuple[str, ...] = (
    "2012-03-25",
    "2013-03-31",
    "2014-03-30",
)
DATAS_DST_OUTUBRO_AFETADAS: tuple[str, ...] = (
    "2012-10-28",
    "2013-10-27",
    "2014-10-26",
)
# No arquivo bruto, o timestamp marca o fim do intervalo de 15 min
HORARIOS_DST_BRUTO: tuple[str, ...] = (
    "01:00:00",
    "01:15:00",
    "01:30:00",
    "01:45:00",
)

# Critérios de tratamento
LIMITE_LACUNA_CURTA_PASSOS: int = 8  # 8 passos * 15 min = 2 horas
INICIO_ESPERADO_COMPLETO: pd.Timestamp = pd.Timestamp("2011-01-01 01:00:00")
FIM_ESPERADO_COMPLETO: pd.Timestamp = pd.Timestamp("2014-12-31 23:00:00")
MAX_PCT_LACUNA_LONGAS: float = 1.0


def carregar_dados_brutos(caminho_txt: Path | str = CAMINHO_BRUTO_PADRAO) -> pd.DataFrame:
    """Carrega o arquivo bruto LD2011_2014.txt e ajusta o índice temporal para o início do intervalo.

    Cada timestamp original representa o fim do intervalo de 15 minutos
    (de 2011-01-01 00:15:00 a 2015-01-01 00:00:00). Subtrai-se 15 minutos para
    alinhar cada leitura ao início do respectivo bloco de medição
    (2011-01-01 00:00:00 a 2014-12-31 23:45:00), garantindo grade regular '15min'.
    """
    caminho = Path(caminho_txt)
    df = pd.read_csv(
        caminho,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
        dtype=np.float64,
        low_memory=False,
    )
    df.index.name = "timestamp_fim"
    df.index = df.index - pd.Timedelta(minutes=15)
    df.index.name = "timestamp"
    # Garante ordenação e frequência explícita caso a grade esteja completa
    df = df.sort_index()
    try:
        df.index.freq = pd.infer_freq(df.index)
    except ValueError:
        pass
    return df


def carregar_serie_tratada(
    caminho_parquet: Path | str,
    freq: str | None = None,
) -> pd.DataFrame:
    """Carrega um arquivo Parquet tratado e restaura a frequência do DatetimeIndex.

    O formato Parquet via pyarrow não preserva `DatetimeIndex.freq` automaticamente;
    esta função garante ordenação temporal e reatribui `freq` explícita para uso
    seguro em reamostragem, decomposição sazonal e modelos ARIMA/SARIMAX.
    """
    df = pd.read_parquet(caminho_parquet)
    df = df.sort_index()
    if freq is not None:
        df = df.asfreq(freq)
    else:
        freq_inferida = pd.infer_freq(df.index)
        if freq_inferida is not None:
            df.index.freq = freq_inferida
    return df


def _mascara_horarios_dst_bruto(indice: pd.DatetimeIndex, datas: tuple[str, ...]) -> np.ndarray:
    """Retorna máscara booleana das linhas correspondentes aos horários DST do arquivo bruto."""
    # Converte o índice ajustado (início do intervalo) de volta ao timestamp original do arquivo
    ts_bruto = indice + pd.Timedelta(minutes=15)
    alvos = pd.DatetimeIndex([f"{d} {h}" for d in datas for h in HORARIOS_DST_BRUTO])
    return ts_bruto.isin(alvos)


def _contar_runs_internos_curtos(
    serie: np.ndarray,
    idx_inicio: int,
    idx_fim: int,
    limite_passos: int,
) -> tuple[np.ndarray, int, int]:
    """Identifica índices de zeros internos curtos (<= limite_passos) dentro [idx_inicio, idx_fim].

    Retorna (indices_globais_zeros_curtos, total_zeros_internos, total_zeros_curtos).
    """
    trecho = serie[idx_inicio : idx_fim + 1]
    eh_zero = trecho == 0.0
    total_zeros = int(eh_zero.sum())
    if total_zeros == 0:
        return np.array([], dtype=np.int64), 0, 0

    # Localiza inícios e fins de blocos contínuos de zeros
    padded = np.concatenate(([False], eh_zero, [False]))
    diff = np.diff(padded.astype(np.int8))
    inicios = np.where(diff == 1)[0]
    fins = np.where(diff == -1)[0]
    comprimentos = fins - inicios

    indices_curtos: list[np.ndarray] = []
    for ini, fim, comp in zip(inicios, fins, comprimentos):
        if comp <= limite_passos:
            indices_curtos.append(np.arange(idx_inicio + ini, idx_inicio + fim, dtype=np.int64))

    if not indices_curtos:
        return np.array([], dtype=np.int64), total_zeros, 0

    concatenados = np.concatenate(indices_curtos)
    return concatenados, total_zeros, int(len(concatenados))


def diagnosticar_dados_brutos(df_bruto: pd.DataFrame) -> dict[str, Any]:
    """Realiza auditoria completa dos dados brutos antes de qualquer modificação."""
    n_linhas, n_colunas = df_bruto.shape
    total_celulas = int(n_linhas * n_colunas)

    nulos_explicitos = int(df_bruto.isna().sum().sum())
    duplicatas_timestamp = int(df_bruto.index.duplicated().sum())
    indice_monotonico = bool(df_bruto.index.is_monotonic_increasing)

    diffs = df_bruto.index.to_series().diff().dropna()
    passos_irregulares = int((diffs != pd.Timedelta(minutes=15)).sum())
    valores_negativos = int((df_bruto < 0).sum().sum())
    zeros_totais = int((df_bruto == 0.0).sum().sum())

    valores = df_bruto.to_numpy()
    nao_zero = valores != 0.0
    possui_ativo = nao_zero.any(axis=0)

    zeros_pre_adesao = 0
    zeros_pos_desligamento = 0
    zeros_internos = 0

    for j in range(n_colunas):
        if not possui_ativo[j]:
            zeros_pre_adesao += n_linhas
            continue
        indices_nz = np.where(nao_zero[:, j])[0]
        ini = int(indices_nz[0])
        fim = int(indices_nz[-1])
        zeros_pre_adesao += ini
        zeros_pos_desligamento += n_linhas - 1 - fim
        zeros_internos += int((valores[ini : fim + 1, j] == 0.0).sum())

    # Estatísticas agregadas nas janelas de DST
    mask_marco = _mascara_horarios_dst_bruto(df_bruto.index, DATAS_DST_MARCO)
    mask_outubro = _mascara_horarios_dst_bruto(df_bruto.index, DATAS_DST_OUTUBRO)

    soma_total = df_bruto.sum(axis=1)
    dst_marco_resumo = {}
    for data in DATAS_DST_MARCO:
        m_data = _mascara_horarios_dst_bruto(df_bruto.index, (data,))
        dst_marco_resumo[data] = {
            "media_soma_kw": float(soma_total[m_data].mean()),
            "media_clientes_nao_zero": float((df_bruto[m_data] != 0.0).sum(axis=1).mean()),
        }

    dst_outubro_resumo = {}
    for data in DATAS_DST_OUTUBRO:
        m_data = _mascara_horarios_dst_bruto(df_bruto.index, (data,))
        dst_outubro_resumo[data] = {
            "media_soma_kw": float(soma_total[m_data].mean()),
            "media_clientes_nao_zero": float((df_bruto[m_data] != 0.0).sum(axis=1).mean()),
        }

    return {
        "n_linhas": int(n_linhas),
        "n_colunas": int(n_colunas),
        "total_celulas": total_celulas,
        "inicio_indice": str(df_bruto.index.min()),
        "fim_indice": str(df_bruto.index.max()),
        "frequencia_inferida": str(df_bruto.index.freqstr),
        "nulos_explicitos": nulos_explicitos,
        "duplicatas_timestamp": duplicatas_timestamp,
        "indice_monotonico_crescente": indice_monotonico,
        "passos_irregulares_15min": passos_irregulares,
        "valores_negativos": valores_negativos,
        "zeros_totais": zeros_totais,
        "pct_zeros_totais": round(100.0 * zeros_totais / total_celulas, 4),
        "zeros_pre_adesao": int(zeros_pre_adesao),
        "pct_zeros_pre_adesao": round(100.0 * zeros_pre_adesao / total_celulas, 4),
        "zeros_pos_desligamento": int(zeros_pos_desligamento),
        "pct_zeros_pos_desligamento": round(100.0 * zeros_pos_desligamento / total_celulas, 4),
        "zeros_internos": int(zeros_internos),
        "pct_zeros_internos": round(100.0 * zeros_internos / total_celulas, 4),
        "linhas_dst_marco": int(mask_marco.sum()),
        "linhas_dst_outubro": int(mask_outubro.sum()),
        "dst_marco_resumo": dst_marco_resumo,
        "dst_outubro_resumo": dst_outubro_resumo,
    }


def tratar_series_15min(
    df_bruto: pd.DataFrame,
    limite_lacuna_curta: int = LIMITE_LACUNA_CURTA_PASSOS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Executa limpeza, correção de DST e interpolação de falhas curtas na grade de 15 min.

    Etapas aplicadas sobre cópia dos dados brutos:
    A) Zeros pré-adesão e pós-desligamento convertidos para NaN.
    B) Correção do DST de março (2012–2014): toda a janela da hora saltada em clientes
       ativos é convertida para NaN (remove zeros e resíduos espúrios).
    C) Correção do DST de outubro (2012–2014): toda a janela da hora repetida em clientes
       ativos é dividida por 2.0 (2011 não apresenta duplicação e é preservado).
    D) Zeros internos curtos (<= limite_lacuna_curta passos consecutivos) convertidos
       em NaN e interpolados por tempo junto às lacunas de DST de março.
    """
    indice = df_bruto.index
    colunas = list(df_bruto.columns)
    n_linhas, n_colunas = df_bruto.shape
    valores = df_bruto.to_numpy(dtype=np.float64, copy=True)

    # Localiza início e fim de operação de cada cliente antes de alterar a matriz
    inicio_idx = np.zeros(n_colunas, dtype=np.int64)
    fim_idx = np.zeros(n_colunas, dtype=np.int64)
    for j in range(n_colunas):
        nz = np.where(valores[:, j] != 0.0)[0]
        if len(nz) == 0:
            inicio_idx[j] = n_linhas
            fim_idx[j] = -1
        else:
            inicio_idx[j] = int(nz[0])
            fim_idx[j] = int(nz[-1])

    # Etapa A: Zeros pré-adesão e pós-desligamento -> NaN
    celulas_pre_adesao = 0
    celulas_pos_desligamento = 0
    for j in range(n_colunas):
        ini = int(inicio_idx[j])
        fim = int(fim_idx[j])
        if ini > 0:
            valores[:ini, j] = np.nan
            celulas_pre_adesao += ini
        if fim < n_linhas - 1:
            valores[fim + 1 :, j] = np.nan
            celulas_pos_desligamento += n_linhas - 1 - fim

    # Etapa B: Correção do DST de março (2012–2014): toda a janela em clientes ativos -> NaN
    mask_marco = _mascara_horarios_dst_bruto(indice, DATAS_DST_MARCO_AFETADAS)
    pos_marco = np.where(mask_marco)[0]
    celulas_dst_marco_marcadas = 0
    for r in pos_marco:
        # Remove tanto zeros quanto resíduos espúrios (~1,4% do normal) em clientes ativos
        alvos = ~np.isnan(valores[r, :])
        n_alvos = int(alvos.sum())
        if n_alvos > 0:
            valores[r, alvos] = np.nan
            celulas_dst_marco_marcadas += n_alvos

    # Etapa C: Correção do DST de outubro (2012–2014): hora repetida em clientes ativos -> / 2.0
    celulas_dst_outubro_corrigidas = 0
    eventos_dst_outubro_por_ano: dict[str, int] = {}
    for data in DATAS_DST_OUTUBRO_AFETADAS:
        pos_janela = np.where(_mascara_horarios_dst_bruto(indice, (data,)))[0]
        if len(pos_janela) != 4:
            continue
        p0, p1 = int(pos_janela[0]), int(pos_janela[-1]) + 1

        clientes_corrigidos_ano = 0
        for j in range(n_colunas):
            bloco = valores[p0:p1, j]
            if np.isnan(bloco).any():
                continue
            valores[p0:p1, j] = bloco / 2.0
            celulas_dst_outubro_corrigidas += 4
            clientes_corrigidos_ano += 1
        eventos_dst_outubro_por_ano[data] = clientes_corrigidos_ano

    # Etapa D: Detectar zeros internos curtos (falhas curtas de telemetria) -> NaN
    celulas_zeros_internos_curtos = 0
    zeros_internos_longos_por_cliente = np.zeros(n_colunas, dtype=np.int64)
    zeros_internos_curtos_por_cliente = np.zeros(n_colunas, dtype=np.int64)

    for j in range(n_colunas):
        ini = int(inicio_idx[j])
        fim = int(fim_idx[j])
        if ini > fim:
            continue
        idx_curtos, total_z_int, total_curtos = _contar_runs_internos_curtos(
            valores[:, j], ini, fim, limite_lacuna_curta
        )
        if len(idx_curtos) > 0:
            valores[idx_curtos, j] = np.nan
            celulas_zeros_internos_curtos += len(idx_curtos)
        zeros_internos_curtos_por_cliente[j] = total_curtos
        zeros_internos_longos_por_cliente[j] = total_z_int - total_curtos

    df_limpo = pd.DataFrame(valores, index=indice, columns=colunas)
    nans_antes_interpolacao = int(df_limpo.isna().sum().sum())

    # Interpolação temporal estritamente limitada a lacunas curtas internas
    df_tratado = df_limpo.interpolate(
        method="time",
        limit=limite_lacuna_curta,
        limit_area="inside",
    )
    nans_apos_interpolacao = int(df_tratado.isna().sum().sum())
    celulas_interpoladas = nans_antes_interpolacao - nans_apos_interpolacao

    # Resumo detalhado por cliente
    registros_clientes: list[dict[str, Any]] = []
    for j, col in enumerate(colunas):
        ini = int(inicio_idx[j])
        fim = int(fim_idx[j])
        passos_ativos = fim - ini + 1 if fim >= ini else 0
        cobertura_bruta_pct = 100.0 * passos_ativos / n_linhas
        s_tratada = df_tratado.iloc[:, j]
        nans_finais_col = int(s_tratada.isna().sum())
        nans_internos_finais = int(s_tratada.iloc[ini : fim + 1].isna().sum()) if passos_ativos > 0 else 0
        z_longos = int(zeros_internos_longos_por_cliente[j])
        pct_z_longos_ativo = (100.0 * z_longos / passos_ativos) if passos_ativos > 0 else 0.0

        registros_clientes.append(
            {
                "cliente": col,
                "inicio_ativo": str(indice[ini]) if fim >= ini else None,
                "fim_ativo": str(indice[fim]) if fim >= ini else None,
                "passos_15min_ativos": passos_ativos,
                "cobertura_temporal_pct": round(cobertura_bruta_pct, 4),
                "zeros_internos_curtos_imputados": int(zeros_internos_curtos_por_cliente[j]),
                "zeros_internos_longos_preservados": z_longos,
                "pct_zeros_longos_na_janela_ativa": round(pct_z_longos_ativo, 4),
                "nans_finais_15min": nans_finais_col,
                "nans_internos_finais": nans_internos_finais,
                "media_kw_tratada": round(float(s_tratada.mean()), 4),
                "desvio_kw_tratada": round(float(s_tratada.std()), 4),
                "max_kw_tratada": round(float(s_tratada.max()), 4),
            }
        )

    df_resumo_clientes = pd.DataFrame(registros_clientes)

    total_celulas = int(n_linhas * n_colunas)
    resumo_tratamento = {
        "celulas_pre_adesao_nan": int(celulas_pre_adesao),
        "pct_pre_adesao_nan": round(100.0 * celulas_pre_adesao / total_celulas, 4),
        "celulas_pos_desligamento_nan": int(celulas_pos_desligamento),
        "pct_pos_desligamento_nan": round(100.0 * celulas_pos_desligamento / total_celulas, 4),
        "celulas_dst_marco_identificadas": int(celulas_dst_marco_marcadas),
        "celulas_dst_outubro_divididas_por_2": int(celulas_dst_outubro_corrigidas),
        "eventos_dst_outubro_por_data": eventos_dst_outubro_por_ano,
        "celulas_zeros_internos_curtos_nan": int(celulas_zeros_internos_curtos),
        "nans_antes_interpolacao": int(nans_antes_interpolacao),
        "celulas_interpoladas_total": int(celulas_interpoladas),
        "pct_celulas_interpoladas_sobre_base_total": round(
            100.0 * celulas_interpoladas / total_celulas, 6
        ),
        "nans_remanescentes_15min": int(nans_apos_interpolacao),
        "nans_remanescentes_janela_ativa": int(df_resumo_clientes["nans_internos_finais"].sum()),
    }
    return df_tratado, df_resumo_clientes, resumo_tratamento


def agregar_e_filtrar_series(
    df_tratado_15min: pd.DataFrame,
    df_resumo_clientes: pd.DataFrame,
) -> dict[str, Any]:
    """Reamostra para grades regulares horária ('1h') e diária ('1D') e seleciona clientes completos.

    Como os valores originais estão em kW (potência média em blocos de 15 min),
    a média horária (`resample('1h').mean()`) preserva a unidade kW e equivale
    numericamente ao consumo energético horário em kWh/h. Na escala diária,
    calcula-se a potência média diária em kW (`resample('1D').mean()`).
    """
    # Grade horária regular para todos os 370 clientes (média de 4 blocos de 15 min)
    df_horario_370 = df_tratado_15min.resample("1h").mean()
    df_horario_370.index.freq = "h"

    # Critério de seleção para o subconjunto completo 2011–2014
    inicio_ts = pd.to_datetime(df_resumo_clientes["inicio_ativo"])
    fim_ts = pd.to_datetime(df_resumo_clientes["fim_ativo"])
    mask_completos = (
        (inicio_ts <= INICIO_ESPERADO_COMPLETO)
        & (fim_ts >= FIM_ESPERADO_COMPLETO)
        & (df_resumo_clientes["nans_internos_finais"] == 0)
        & (df_resumo_clientes["pct_zeros_longos_na_janela_ativa"] <= MAX_PCT_LACUNA_LONGAS)
    )
    # Também registra quantos atendem apenas ao recorte temporal sem o filtro de zeros longos
    mask_janela_2011_2014 = (
        (inicio_ts <= INICIO_ESPERADO_COMPLETO)
        & (fim_ts >= FIM_ESPERADO_COMPLETO)
        & (df_resumo_clientes["nans_internos_finais"] == 0)
    )

    clientes_janela = df_resumo_clientes.loc[mask_janela_2011_2014, "cliente"].tolist()
    clientes_completos = df_resumo_clientes.loc[mask_completos, "cliente"].tolist()

    df_horario_completos = df_horario_370[clientes_completos].copy()
    df_horario_completos.index.freq = "h"

    df_diario_completos = df_horario_completos.resample("1D").mean()
    df_diario_completos.index.freq = "D"

    # Garante preenchimento limpo caso algum bloco horário completo tenha restado NaN (esperado: 0)
    nans_horario_completos = int(df_horario_completos.isna().sum().sum())
    nans_diario_completos = int(df_diario_completos.isna().sum().sum())

    df_sel = df_resumo_clientes.loc[mask_completos]
    zeros_curtos_sel = int(df_sel["zeros_internos_curtos_imputados"].sum())
    max_zeros_curtos_sel = int(df_sel["zeros_internos_curtos_imputados"].max()) if len(df_sel) > 0 else 0

    resumo_agregacao = {
        "n_clientes_janela_2011_2014": len(clientes_janela),
        "n_clientes_completos_selecionados": len(clientes_completos),
        "clientes_excluidos_por_zeros_longos": sorted(
            list(set(clientes_janela) - set(clientes_completos))
        ),
        "zeros_internos_curtos_imputados_nos_selecionados": zeros_curtos_sel,
        "max_zeros_internos_curtos_por_cliente_selecionado": max_zeros_curtos_sel,
        "shape_horario_370": list(df_horario_370.shape),
        "shape_horario_completos": list(df_horario_completos.shape),
        "shape_diario_completos": list(df_diario_completos.shape),
        "freq_horario": str(df_horario_completos.index.freqstr),
        "freq_diario": str(df_diario_completos.index.freqstr),
        "nans_horario_completos": nans_horario_completos,
        "nans_diario_completos": nans_diario_completos,
        "nans_horario_370_por_inatividade": int(df_horario_370.isna().sum().sum()),
    }

    return {
        "df_horario_370": df_horario_370,
        "df_horario_completos": df_horario_completos,
        "df_diario_completos": df_diario_completos,
        "clientes_completos": clientes_completos,
        "resumo_agregacao": resumo_agregacao,
    }


def gerar_figuras_preparacao(
    df_bruto: pd.DataFrame,
    df_tratado_15min: pd.DataFrame,
    df_horario_completos: pd.DataFrame,
    df_diario_completos: pd.DataFrame,
    df_resumo_clientes: pd.DataFrame,
    diretorio_figuras: Path | str = DIRETORIO_FIGURAS_PADRAO,
) -> list[str]:
    """Gera e salva as figuras de diagnóstico antes/depois e visão geral em Documentacao/figuras/."""
    dir_fig = Path(diretorio_figuras)
    dir_fig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})

    arquivos_salvos: list[str] = []

    # ------------------------------------------------------------------
    # Figura 1: Cobertura temporal dos 370 clientes (antes vs. estrutura)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    clientes_ativos_diario = (
        df_tratado_15min.notna().resample("1D").max().sum(axis=1)
    )
    axes[0].plot(clientes_ativos_diario.index, clientes_ativos_diario.values, color="#1f77b4", lw=1.8)
    axes[0].set_title("Clientes ativos ao longo do tempo (2011–2014)")
    axes[0].set_xlabel("Data")
    axes[0].set_ylabel("Número de clientes ativos (de 370)")
    axes[0].set_ylim(0, 390)
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    coberturas = df_resumo_clientes["cobertura_temporal_pct"].values
    axes[1].hist(coberturas, bins=20, color="#2ca02c", edgecolor="white", alpha=0.85)
    axes[1].axvline(99.0, color="#d62728", linestyle="--", lw=1.5, label="Corte de cobertura completa (≥ 99%)")
    axes[1].set_title("Distribuição da cobertura temporal por cliente")
    axes[1].set_xlabel("Cobertura temporal na janela 2011–2014 (%)")
    axes[1].set_ylabel("Quantidade de clientes")
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    p1 = dir_fig / "etapa0_cobertura_clientes.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p1))

    # ------------------------------------------------------------------
    # Figura 2: Correção de DST Antes vs. Depois (Março e Outubro 2014)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    soma_bruta = df_bruto.sum(axis=1) / 1000.0  # em MW
    soma_tratada = df_tratado_15min.sum(axis=1) / 1000.0  # em MW

    # Painel A: DST Março 2014 (2014-03-30 00:00 a 05:00)
    ini_mar, fim_mar = "2014-03-30 00:00:00", "2014-03-30 05:00:00"
    s_b_mar = soma_bruta.loc[ini_mar:fim_mar]
    s_t_mar = soma_tratada.loc[ini_mar:fim_mar]
    axes[0].plot(s_b_mar.index, s_b_mar.values, "o--", color="#d62728", lw=1.6, ms=4, label="Bruto (com zeros de DST)")
    axes[0].plot(s_t_mar.index, s_t_mar.values, "s-", color="#1f77b4", lw=2.0, ms=4, label="Tratado (interpolação temporal)")
    axes[0].set_title("DST Março (2014-03-30): Remoção de zeros artificiais")
    axes[0].set_xlabel("Hora do dia (HH:MM)")
    axes[0].set_ylabel("Carga agregada (MW)")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].legend(loc="lower right")

    # Painel B: DST Outubro 2014 (2014-10-26 00:00 a 05:00)
    ini_out, fim_out = "2014-10-26 00:00:00", "2014-10-26 05:00:00"
    s_b_out = soma_bruta.loc[ini_out:fim_out]
    s_t_out = soma_tratada.loc[ini_out:fim_out]
    axes[1].plot(s_b_out.index, s_b_out.values, "o--", color="#d62728", lw=1.6, ms=4, label="Bruto (hora dupla somada)")
    axes[1].plot(s_t_out.index, s_t_out.values, "s-", color="#1f77b4", lw=2.0, ms=4, label="Tratado (divisão por 2)")
    axes[1].set_title("DST Outubro (2014-10-26): Correção da hora repetida")
    axes[1].set_xlabel("Hora do dia (HH:MM)")
    axes[1].set_ylabel("Carga agregada (MW)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    p2 = dir_fig / "etapa0_correcao_dst_antes_depois.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p2))

    # ------------------------------------------------------------------
    # Figura 3: Visão geral das séries tratadas (tendência e sazonalidades)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2))

    carga_diaria_mw = df_diario_completos.sum(axis=1) / 1000.0
    media_movel_30d = carga_diaria_mw.rolling(30, center=True).mean()
    axes[0].plot(carga_diaria_mw.index, carga_diaria_mw.values, color="#1f77b4", alpha=0.55, lw=1.0, label="Carga média diária (MW)")
    axes[0].plot(media_movel_30d.index, media_movel_30d.values, color="#d62728", lw=2.0, label="Tendência / Média móvel 30 dias")
    axes[0].set_title(
        f"Carga agregada diária dos {df_diario_completos.shape[1]} clientes completos (2011–2014)"
    )
    axes[0].set_xlabel("Data")
    axes[0].set_ylabel("Potência média diária (MW)")
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper left")

    # Painel B: Perfil horário normalizado de 4 clientes distintos ao longo de 1 semana típica
    semana_inicio, semana_fim = "2014-05-05 00:00:00", "2014-05-11 23:00:00"
    medias_clientes = df_horario_completos.mean().sort_values()
    # Escolhe 4 quantis para ilustrar perfis de magnitudes distintas normalizados em Z-score na semana
    indices_q = [
        int(0.10 * len(medias_clientes)),
        int(0.40 * len(medias_clientes)),
        int(0.70 * len(medias_clientes)),
        int(0.95 * len(medias_clientes)),
    ]
    exemplares = [medias_clientes.index[i] for i in indices_q]
    trecho_semana = df_horario_completos.loc[semana_inicio:semana_fim, exemplares]
    for col in exemplares:
        z = (trecho_semana[col] - trecho_semana[col].mean()) / (trecho_semana[col].std() + 1e-9)
        axes[1].plot(z.index, z.values, lw=1.4, label=f"{col} (média={medias_clientes[col]:.1f} kW)")

    axes[1].set_title("Perfil horário padronizado (Z-score) de clientes exemplares — 1 semana (seg a dom)")
    axes[1].set_xlabel("Dia da semana (05/05/2014 a 11/05/2014)")
    axes[1].set_ylabel("Consumo padronizado (Z-score)")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m"))
    axes[1].legend(loc="upper right", ncol=2)

    plt.tight_layout()
    p3 = dir_fig / "etapa0_series_tratadas_visao_geral.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p3))

    return arquivos_salvos


def salvar_saidas_preparacao(
    agregado: dict[str, Any],
    df_resumo_clientes: pd.DataFrame,
    metricas: dict[str, Any],
    dir_tratado: Path | str = DIRETORIO_TRATADO_PADRAO,
    dir_resultados: Path | str = DIRETORIO_RESULTADOS_PADRAO,
) -> dict[str, str]:
    """Persiste os datasets tratados em Parquet e os relatórios quantitativos em CSV/JSON."""
    p_tratado = Path(dir_tratado)
    p_resultados = Path(dir_resultados)
    p_tratado.mkdir(parents=True, exist_ok=True)
    p_resultados.mkdir(parents=True, exist_ok=True)

    arq_h_comp = p_tratado / "consumo_horario_clientes_completos_2011_2014.parquet"
    arq_d_comp = p_tratado / "consumo_diario_clientes_completos_2011_2014.parquet"
    arq_h_370 = p_tratado / "consumo_horario_370_clientes.parquet"
    arq_resumo_cli = p_resultados / "resumo_clientes_preparacao.csv"
    arq_metricas = p_resultados / "metricas_preparacao.json"

    agregado["df_horario_completos"].to_parquet(arq_h_comp)
    agregado["df_diario_completos"].to_parquet(arq_d_comp)
    agregado["df_horario_370"].to_parquet(arq_h_370)
    df_resumo_clientes.to_csv(arq_resumo_cli, index=False)

    with open(arq_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    return {
        "horario_completos": str(arq_h_comp),
        "diario_completos": str(arq_d_comp),
        "horario_370": str(arq_h_370),
        "resumo_clientes_csv": str(arq_resumo_cli),
        "metricas_json": str(arq_metricas),
    }
