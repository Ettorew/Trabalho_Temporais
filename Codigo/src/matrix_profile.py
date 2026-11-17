"""Módulo de descoberta de padrões e anomalias (Etapa 2): Matrix Profile com STUMPY."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy

# Semente padrão e hiperparâmetros explícitos da Etapa 2
SEMENTE_PADRAO: int = 42
CLIENTE_REPRESENTATIVO: str = "MT_206"
JANELA_HORARIA_DIARIA_M: int = 24     # 24 horas = 1 ciclo diário
JANELA_HORARIA_SEMANAL_M: int = 168   # 168 horas = 7 dias (1 ciclo semanal completo)
JANELA_DIARIA_SEMANAL_M: int = 7      # 7 dias = 1 semana na escala diária
TOP_K_PADROES: int = 3

# Parâmetros do experimento controlado de invariância à escala e offset (Z-score)
FATOR_ESCALA_SINTETICO: float = 1.8
OFFSET_KW_SINTETICO: float = 150.0

# Caminhos padrão relativos à raiz do repositório
CAMINHO_HORARIO_PADRAO: Path = Path("Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet")
CAMINHO_DIARIO_PADRAO: Path = Path("Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet")
DIRETORIO_RESULTADOS_PADRAO: Path = Path("Resultados")
DIRETORIO_FIGURAS_PADRAO: Path = Path("Documentacao/figuras")


def _zscore_1d(vetor: np.ndarray) -> np.ndarray:
    """Normaliza um vetor 1D para média zero e desvio-padrão unitário."""
    media = float(np.mean(vetor))
    desvio = float(np.std(vetor, ddof=0))
    if desvio <= 1e-12:
        return np.zeros_like(vetor, dtype=np.float64)
    return (vetor - media) / desvio


def calcular_ou_carregar_mp(
    serie: pd.Series,
    m: int,
    normalizar: bool = True,
    caminho_cache: Path | str | None = None,
    forcar_recalculo: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Calcula ou carrega em cache o Matrix Profile (distância e índice NN) via `stumpy.stump`."""
    valores = serie.to_numpy(dtype=np.float64)
    n_pontos = int(len(valores))
    n_subseq = n_pontos - m + 1
    if n_subseq <= 0:
        raise ValueError(f"Série de tamanho {n_pontos} é insuficiente para janela m={m}.")

    carregado_do_cache = False
    tempo_calculo_seg = 0.0
    matriz_mp: np.ndarray | None = None

    if caminho_cache is not None:
        caminho = Path(caminho_cache)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        if caminho.exists() and not forcar_recalculo:
            arr_cache = np.load(caminho)
            if arr_cache.shape == (n_subseq, 2):
                matriz_mp = arr_cache
                carregado_do_cache = True
    else:
        caminho = None

    if matriz_mp is None:
        t0 = time.perf_counter()
        mp = stumpy.stump(valores, m=m, normalize=normalizar)
        tempo_calculo_seg = float(time.perf_counter() - t0)
        mp_dist = np.asarray(mp[:, 0], dtype=np.float64)
        mp_idx = np.asarray(mp[:, 1], dtype=np.int64)
        matriz_mp = np.column_stack([mp_dist, mp_idx.astype(np.float64)])
        if caminho is not None:
            np.save(caminho, matriz_mp)
    else:
        mp_dist = np.asarray(matriz_mp[:, 0], dtype=np.float64)
        mp_idx = np.asarray(matriz_mp[:, 1], dtype=np.int64)

    finitos = mp_dist[np.isfinite(mp_dist)]
    estatisticas = {
        "m": int(m),
        "normalizar_zscore": bool(normalizar),
        "n_pontos_serie": n_pontos,
        "n_subsequencias": int(n_subseq),
        "zona_exclusao_padrao": int(np.ceil(m / 4.0)),
        "carregado_do_cache": carregado_do_cache,
        "tempo_calculo_seg": round(tempo_calculo_seg, 3),
        "caminho_cache": str(caminho) if caminho is not None else None,
        "dist_min": round(float(np.min(finitos)), 4),
        "dist_p05": round(float(np.percentile(finitos, 5)), 4),
        "dist_mediana": round(float(np.median(finitos)), 4),
        "dist_media": round(float(np.mean(finitos)), 4),
        "dist_p95": round(float(np.percentile(finitos, 95)), 4),
        "dist_p99": round(float(np.percentile(finitos, 99)), 4),
        "dist_max": round(float(np.max(finitos)), 4),
    }
    return mp_dist, mp_idx, estatisticas


def extrair_top_motifs(
    serie: pd.Series,
    mp_dist: np.ndarray,
    mp_idx: np.ndarray,
    m: int,
    top_k: int = TOP_K_PADROES,
) -> list[dict[str, Any]]:
    """Extrai os top_k pares de motifs não sobrepostos (menores distâncias no MP)."""
    valores = serie.to_numpy(dtype=np.float64)
    indice = serie.index
    dists = mp_dist.astype(np.float64, copy=True)
    dists[~np.isfinite(dists)] = np.inf
    n_sub = len(dists)

    motifs: list[dict[str, Any]] = []
    for rank in range(1, top_k + 1):
        if not np.isfinite(dists).any():
            break
        idx_a = int(np.argmin(dists))
        dist_val = float(dists[idx_a])
        if not np.isfinite(dist_val):
            break
        idx_b = int(mp_idx[idx_a])

        sub_a = valores[idx_a : idx_a + m]
        sub_b = valores[idx_b : idx_b + m]
        corr_pearson = float(np.corrcoef(sub_a, sub_b)[0, 1])
        corr_teorica = float(1.0 - (dist_val ** 2) / (2.0 * m))

        media_a = float(np.mean(sub_a))
        media_b = float(np.mean(sub_b))
        desvio_a = float(np.std(sub_a, ddof=0))
        desvio_b = float(np.std(sub_b, ddof=0))

        motifs.append(
            {
                "tipo": "motif",
                "rank": rank,
                "m": int(m),
                "idx_a": idx_a,
                "idx_b": idx_b,
                "inicio_a": str(indice[idx_a]),
                "fim_a": str(indice[idx_a + m - 1]),
                "inicio_b": str(indice[idx_b]),
                "fim_b": str(indice[idx_b + m - 1]),
                "distancia_mp": round(dist_val, 4),
                "correlacao_pearson": round(corr_pearson, 4),
                "correlacao_teorica_z": round(corr_teorica, 4),
                "media_kw_a": round(media_a, 2),
                "media_kw_b": round(media_b, 2),
                "diferenca_media_kw": round(abs(media_a - media_b), 2),
                "desvio_kw_a": round(desvio_a, 2),
                "desvio_kw_b": round(desvio_b, 2),
            }
        )

        # Zona de exclusão de raio m em torno de ambos os membros para evitar deslocamentos triviais
        dists[max(0, idx_a - m) : min(n_sub, idx_a + m + 1)] = np.inf
        dists[max(0, idx_b - m) : min(n_sub, idx_b + m + 1)] = np.inf

    return motifs


def extrair_top_discords(
    serie: pd.Series,
    mp_dist: np.ndarray,
    mp_idx: np.ndarray,
    m: int,
    top_k: int = TOP_K_PADROES,
) -> list[dict[str, Any]]:
    """Extrai as top_k subsequências anômalas (discords) não sobrepostas."""
    valores = serie.to_numpy(dtype=np.float64)
    indice = serie.index
    dists = mp_dist.astype(np.float64, copy=True)
    dists[~np.isfinite(dists)] = -np.inf
    n_sub = len(dists)

    discords: list[dict[str, Any]] = []
    for rank in range(1, top_k + 1):
        if not np.isfinite(dists).any():
            break
        idx_d = int(np.argmax(dists))
        dist_val = float(dists[idx_d])
        if not np.isfinite(dist_val):
            break
        idx_v = int(mp_idx[idx_d])

        sub_d = valores[idx_d : idx_d + m]
        sub_v = valores[idx_v : idx_v + m]
        corr_viz = float(np.corrcoef(sub_d, sub_v)[0, 1])

        discords.append(
            {
                "tipo": "discord",
                "rank": rank,
                "m": int(m),
                "idx_discord": idx_d,
                "idx_vizinho": idx_v,
                "inicio_discord": str(indice[idx_d]),
                "fim_discord": str(indice[idx_d + m - 1]),
                "inicio_vizinho": str(indice[idx_v]),
                "fim_vizinho": str(indice[idx_v + m - 1]),
                "distancia_mp": round(dist_val, 4),
                "correlacao_com_vizinho": round(corr_viz, 4),
                "media_kw_discord": round(float(np.mean(sub_d)), 2),
                "desvio_kw_discord": round(float(np.std(sub_d, ddof=0)), 2),
                "min_kw_discord": round(float(np.min(sub_d)), 2),
                "max_kw_discord": round(float(np.max(sub_d)), 2),
                "media_kw_vizinho": round(float(np.mean(sub_v)), 2),
                "desvio_kw_vizinho": round(float(np.std(sub_v, ddof=0)), 2),
            }
        )

        # Zona de exclusão de raio m para destacar eventos anômalos distintos no tempo
        dists[max(0, idx_d - m) : min(n_sub, idx_d + m + 1)] = -np.inf

    return discords


def executar_experimento_zscore(
    serie_horaria: pd.Series,
    m: int = JANELA_HORARIA_SEMANAL_M,
    idx_alvo: int | None = None,
    fator_escala: float = FATOR_ESCALA_SINTETICO,
    offset_kw: float = OFFSET_KW_SINTETICO,
    mp_norm_precomp: tuple[np.ndarray, np.ndarray] | None = None,
    caminho_cache_sem_norm: Path | str | None = None,
) -> dict[str, Any]:
    """Compara MP com Z-score vs. sem normalização na série real e sob perturbação afim controlada."""
    if mp_norm_precomp is None:
        mp_norm_dist, mp_norm_idx, _ = calcular_ou_carregar_mp(
            serie_horaria, m=m, normalizar=True
        )
    else:
        mp_norm_dist, mp_norm_idx = mp_norm_precomp

    mp_sem_dist, mp_sem_idx, est_sem = calcular_ou_carregar_mp(
        serie_horaria,
        m=m,
        normalizar=False,
        caminho_cache=caminho_cache_sem_norm,
    )

    # Desvio-padrão e média locais de cada subsequência de comprimento m
    std_local = serie_horaria.rolling(m).std(ddof=0).dropna().to_numpy(dtype=np.float64)
    mean_local = serie_horaria.rolling(m).mean().dropna().to_numpy(dtype=np.float64)

    corr_sem_std = float(np.corrcoef(mp_sem_dist, std_local)[0, 1])
    corr_sem_mean = float(np.corrcoef(mp_sem_dist, mean_local)[0, 1])
    corr_norm_std = float(np.corrcoef(mp_norm_dist, std_local)[0, 1])
    corr_norm_mean = float(np.corrcoef(mp_norm_dist, mean_local)[0, 1])

    top_disc_sem = extrair_top_discords(serie_horaria, mp_sem_dist, mp_sem_idx, m=m, top_k=3)
    top_mot_sem = extrair_top_motifs(serie_horaria, mp_sem_dist, mp_sem_idx, m=m, top_k=1)

    if idx_alvo is None:
        idx_alvo = int(np.argmin(mp_norm_dist))

    # Experimento controlado: perturbação afim pura X' = fator * X + offset na janela idx_alvo
    valores_pert = serie_horaria.to_numpy(dtype=np.float64, copy=True)
    valores_pert[idx_alvo : idx_alvo + m] = (
        fator_escala * valores_pert[idx_alvo : idx_alvo + m] + offset_kw
    )
    serie_pert = pd.Series(valores_pert, index=serie_horaria.index, name=serie_horaria.name)

    mp_pert_norm = stumpy.stump(valores_pert, m=m, normalize=True)
    mp_pert_sem = stumpy.stump(valores_pert, m=m, normalize=False)

    pert_norm_dist = np.asarray(mp_pert_norm[:, 0], dtype=np.float64)
    pert_sem_dist = np.asarray(mp_pert_sem[:, 0], dtype=np.float64)

    dist_orig_norm_alvo = float(mp_norm_dist[idx_alvo])
    dist_pert_norm_alvo = float(pert_norm_dist[idx_alvo])
    dist_orig_sem_alvo = float(mp_sem_dist[idx_alvo])
    dist_pert_sem_alvo = float(pert_sem_dist[idx_alvo])

    # Posição relativa (rank 1 = maior distância; rank N = menor distância)
    n_sub = len(pert_norm_dist)
    rank_disc_pert_norm = int((pert_norm_dist > dist_pert_norm_alvo).sum() + 1)
    rank_disc_pert_sem = int((pert_sem_dist > dist_pert_sem_alvo).sum() + 1)
    rank_mot_pert_norm = int((pert_norm_dist < dist_pert_norm_alvo).sum() + 1)

    return {
        "serie_original": {
            "m": int(m),
            "tempo_calculo_sem_norm_seg": est_sem["tempo_calculo_seg"],
            "corr_mp_sem_norm_com_desvio_local": round(corr_sem_std, 4),
            "corr_mp_sem_norm_com_media_local": round(corr_sem_mean, 4),
            "corr_mp_zscore_com_desvio_local": round(corr_norm_std, 4),
            "corr_mp_zscore_com_media_local": round(corr_norm_mean, 4),
            "top1_motif_sem_norm": top_mot_sem[0] if top_mot_sem else None,
            "top3_discords_sem_norm": top_disc_sem,
            "mp_sem_dist": mp_sem_dist,
            "std_local": std_local,
        },
        "experimento_perturbacao": {
            "idx_alvo": int(idx_alvo),
            "inicio_alvo": str(serie_horaria.index[idx_alvo]),
            "fim_alvo": str(serie_horaria.index[idx_alvo + m - 1]),
            "fator_escala": float(fator_escala),
            "offset_kw": float(offset_kw),
            "dist_orig_zscore": round(dist_orig_norm_alvo, 4),
            "dist_pert_zscore": round(dist_pert_norm_alvo, 4),
            "delta_abs_zscore": round(abs(dist_pert_norm_alvo - dist_orig_norm_alvo), 6),
            "rank_motif_pert_zscore": rank_mot_pert_norm,
            "rank_discord_pert_zscore": rank_disc_pert_norm,
            "dist_orig_sem_norm": round(dist_orig_sem_alvo, 2),
            "dist_pert_sem_norm": round(dist_pert_sem_alvo, 2),
            "razao_salto_sem_norm": round(dist_pert_sem_alvo / max(dist_orig_sem_alvo, 1e-9), 2),
            "rank_discord_pert_sem_norm": rank_disc_pert_sem,
            "idx_max_discord_pert_sem_norm": int(np.argmax(pert_sem_dist)),
            "dist_max_discord_pert_sem_norm": round(float(np.max(pert_sem_dist)), 2),
            "n_subsequencias": int(n_sub),
            "serie_pert": serie_pert,
            "pert_norm_dist": pert_norm_dist,
            "pert_sem_dist": pert_sem_dist,
        },
    }


def _perfil_semanal_mediano_z(serie_horaria: pd.Series) -> np.ndarray:
    """Calcula a forma semanal mediana (168h, seg 00h -> dom 23h) em Z-score da série."""
    df_tmp = pd.DataFrame(
        {
            "kw": serie_horaria.values,
            "hora_semana": serie_horaria.index.dayofweek * 24 + serie_horaria.index.hour,
        }
    )
    mediana_168 = df_tmp.groupby("hora_semana")["kw"].median().to_numpy(dtype=np.float64)
    return _zscore_1d(mediana_168)


def gerar_figuras_matrix_profile(
    serie_horaria: pd.Series,
    serie_diaria: pd.Series,
    mp_168: tuple[np.ndarray, np.ndarray],
    mp_24: tuple[np.ndarray, np.ndarray],
    mp_7d: tuple[np.ndarray, np.ndarray],
    motifs_168: list[dict[str, Any]],
    motifs_24: list[dict[str, Any]],
    motifs_7d: list[dict[str, Any]],
    discords_168: list[dict[str, Any]],
    discords_24: list[dict[str, Any]],
    exp_zscore: dict[str, Any],
    diretorio_figuras: Path | str = DIRETORIO_FIGURAS_PADRAO,
) -> list[str]:
    """Gera e salva as 4 figuras diagnósticas da Etapa 2 em Documentacao/figuras/."""
    dir_fig = Path(diretorio_figuras)
    dir_fig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})

    arquivos_salvos: list[str] = []
    mp168_dist, _ = mp_168
    mp24_dist, _ = mp_24
    mp7d_dist, _ = mp_7d

    m168 = JANELA_HORARIA_SEMANAL_M
    m24 = JANELA_HORARIA_DIARIA_M
    m7 = JANELA_DIARIA_SEMANAL_M

    mot1_168 = motifs_168[0]
    disc1_168 = discords_168[0]
    disc2_168 = discords_168[1]

    # ------------------------------------------------------------------
    # Figura 8 (etapa2_mp_visao_geral_comparacao_m.png)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(14.5, 11.2), sharex=True)

    axes[0].plot(serie_horaria.index, serie_horaria.values, color="#1f77b4", lw=0.45, alpha=0.85, label="Consumo horário MT_206 (kW)")
    # Destaca Motif #1 (a e b) e Discord #1 (m=168h)
    axes[0].axvspan(pd.Timestamp(mot1_168["inicio_a"]), pd.Timestamp(mot1_168["fim_a"]), color="#2ca02c", alpha=0.35, label="Motif #1 (par A e B, m=168h)")
    axes[0].axvspan(pd.Timestamp(mot1_168["inicio_b"]), pd.Timestamp(mot1_168["fim_b"]), color="#2ca02c", alpha=0.35)
    axes[0].axvspan(pd.Timestamp(disc1_168["inicio_discord"]), pd.Timestamp(disc1_168["fim_discord"]), color="#d62728", alpha=0.35, label="Discord #1 (m=168h)")
    axes[0].set_title(f"Série Representativa {serie_horaria.name} (2011–2014) e Matrix Profiles Multi-Escala (Z-score)")
    axes[0].set_ylabel("Consumo (kW)")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8.5)

    idx_mp168 = serie_horaria.index[: len(mp168_dist)]
    axes[1].plot(idx_mp168, mp168_dist, color="#1f77b4", lw=0.75)
    axes[1].scatter(
        [pd.Timestamp(mot1_168["inicio_a"]), pd.Timestamp(mot1_168["inicio_b"])],
        [mp168_dist[mot1_168["idx_a"]], mp168_dist[mot1_168["idx_b"]]],
        color="#2ca02c", s=45, zorder=5, label=f"Motif #1 (d={mot1_168['distancia_mp']:.2f})",
    )
    axes[1].scatter(
        [pd.Timestamp(disc1_168["inicio_discord"])],
        [disc1_168["distancia_mp"]],
        color="#d62728", s=55, marker="X", zorder=5, label=f"Discord #1 (d={disc1_168['distancia_mp']:.2f})",
    )
    axes[1].set_title("Matrix Profile Horário — Janela Semanal m = 168 h (Principal)")
    axes[1].set_ylabel("Distância MP ($d_Z$)")
    axes[1].legend(loc="upper right", fontsize=8.5)

    idx_mp24 = serie_horaria.index[: len(mp24_dist)]
    mot1_24 = motifs_24[0]
    disc1_24 = discords_24[0]
    axes[2].plot(idx_mp24, mp24_dist, color="#ff7f0e", lw=0.65)
    axes[2].scatter(
        [pd.Timestamp(mot1_24["inicio_a"]), pd.Timestamp(mot1_24["inicio_b"])],
        [mp24_dist[mot1_24["idx_a"]], mp24_dist[mot1_24["idx_b"]]],
        color="#2ca02c", s=40, zorder=5, label=f"Motif #1 (d={mot1_24['distancia_mp']:.2f})",
    )
    axes[2].scatter(
        [pd.Timestamp(disc1_24["inicio_discord"])],
        [disc1_24["distancia_mp"]],
        color="#d62728", s=50, marker="X", zorder=5, label=f"Discord #1 (d={disc1_24['distancia_mp']:.2f})",
    )
    axes[2].set_title("Matrix Profile Horário — Janela Diária m = 24 h (Ciclo Intradiário)")
    axes[2].set_ylabel("Distância MP ($d_Z$)")
    axes[2].legend(loc="upper right", fontsize=8.5)

    idx_mp7d = serie_diaria.index[: len(mp7d_dist)]
    mot1_7d = motifs_7d[0]
    disc1_7d_idx = int(np.argmax(mp7d_dist))
    axes[3].plot(idx_mp7d, mp7d_dist, color="#9467bd", lw=1.1)
    axes[3].scatter(
        [pd.Timestamp(mot1_7d["inicio_a"]), pd.Timestamp(mot1_7d["inicio_b"])],
        [mp7d_dist[mot1_7d["idx_a"]], mp7d_dist[mot1_7d["idx_b"]]],
        color="#2ca02c", s=45, zorder=5, label=f"Motif #1 (d={mot1_7d['distancia_mp']:.2f})",
    )
    axes[3].scatter(
        [idx_mp7d[disc1_7d_idx]],
        [mp7d_dist[disc1_7d_idx]],
        color="#d62728", s=55, marker="X", zorder=5, label=f"Discord #1 (d={mp7d_dist[disc1_7d_idx]:.2f})",
    )
    axes[3].set_title("Matrix Profile Diário — Janela Semanal m = 7 dias (Escala Agregada)")
    axes[3].set_ylabel("Distância MP ($d_Z$)")
    axes[3].set_xlabel("Data (2011–2014)")
    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[3].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    p8 = dir_fig / "etapa2_mp_visao_geral_comparacao_m.png"
    fig.savefig(p8, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p8))

    # ------------------------------------------------------------------
    # Figura 9 (etapa2_motifs_sobrepostos.png): Motifs sobrepostos
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    sub_a_168 = serie_horaria.iloc[mot1_168["idx_a"] : mot1_168["idx_a"] + m168].to_numpy()
    sub_b_168 = serie_horaria.iloc[mot1_168["idx_b"] : mot1_168["idx_b"] + m168].to_numpy()
    h168 = np.arange(m168)

    # Topo esquerdo: Motif #1 semanal em Z-score
    axes[0, 0].plot(h168, _zscore_1d(sub_a_168), color="#1f77b4", lw=1.7, label=f"A: {mot1_168['inicio_a'][:13]}h (μ={mot1_168['media_kw_a']:.1f} kW)")
    axes[0, 0].plot(h168, _zscore_1d(sub_b_168), color="#d62728", lw=1.5, ls="--", label=f"B: {mot1_168['inicio_b'][:13]}h (μ={mot1_168['media_kw_b']:.1f} kW)")
    for dia_div in range(24, m168, 24):
        axes[0, 0].axvline(dia_div, color="#888888", lw=0.6, ls=":")
    axes[0, 0].set_title(f"Motif #1 Semanal (m=168h) em Z-score | $d_Z$={mot1_168['distancia_mp']:.2f}, r={mot1_168['correlacao_pearson']:.4f}")
    axes[0, 0].set_xlabel("Horas desde o início da janela (divisórias de 24h)")
    axes[0, 0].set_ylabel("Consumo padronizado (Z-score)")
    axes[0, 0].set_xticks(np.arange(0, m168 + 1, 24))
    axes[0, 0].legend(loc="lower right", fontsize=8.3)

    # Topo direito: Mesmo Motif #1 semanal em kW originais
    axes[0, 1].plot(h168, sub_a_168, color="#1f77b4", lw=1.7, label=f"A: {mot1_168['inicio_a'][:10]} ({pd.Timestamp(mot1_168['inicio_a']).day_name()[:3]})")
    axes[0, 1].plot(h168, sub_b_168, color="#d62728", lw=1.5, ls="--", label=f"B: {mot1_168['inicio_b'][:10]} ({pd.Timestamp(mot1_168['inicio_b']).day_name()[:3]})")
    for dia_div in range(24, m168, 24):
        axes[0, 1].axvline(dia_div, color="#888888", lw=0.6, ls=":")
    axes[0, 1].set_title(f"Motif #1 Semanal (m=168h) em kW Originais | Δμ={mot1_168['diferenca_media_kw']:.1f} kW")
    axes[0, 1].set_xlabel("Horas desde o início da janela (divisórias de 24h)")
    axes[0, 1].set_ylabel("Potência (kW)")
    axes[0, 1].set_xticks(np.arange(0, m168 + 1, 24))
    axes[0, 1].legend(loc="lower right", fontsize=8.3)

    # Baixo esquerdo: Primeiro motif diário orgânico (m=24h, d_Z > 0) em Z-score
    # Evita plotar os pares d_Z = 0.0 causados por dias replicados no arquivo bruto original
    mot_org_24 = next((m for m in motifs_24 if m["distancia_mp"] > 1e-6), mot1_24)
    sub_a_24 = serie_horaria.iloc[mot_org_24["idx_a"] : mot_org_24["idx_a"] + m24].to_numpy()
    sub_b_24 = serie_horaria.iloc[mot_org_24["idx_b"] : mot_org_24["idx_b"] + m24].to_numpy()
    h24 = np.arange(m24)
    axes[1, 0].plot(h24, _zscore_1d(sub_a_24), "o-", color="#1f77b4", ms=3.5, lw=1.7, label=f"A: {mot_org_24['inicio_a'][:13]}h (μ={mot_org_24['media_kw_a']:.1f} kW)")
    axes[1, 0].plot(h24, _zscore_1d(sub_b_24), "s--", color="#d62728", ms=3.5, lw=1.5, label=f"B: {mot_org_24['inicio_b'][:13]}h (μ={mot_org_24['media_kw_b']:.1f} kW)")
    axes[1, 0].set_title(
        f"Motif #{mot_org_24['rank']} Intradiário Orgânico (m=24h) | $d_Z$={mot_org_24['distancia_mp']:.4f}, r={mot_org_24['correlacao_pearson']:.4f}"
    )
    axes[1, 0].set_xlabel("Passo dentro da janela de 24h")
    axes[1, 0].set_ylabel("Consumo padronizado (Z-score)")
    axes[1, 0].set_xticks(np.arange(0, m24, 2))
    axes[1, 0].legend(loc="lower right", fontsize=8.3)

    # Baixo direito: Motif #1 na escala diária (m=7 dias) em Z-score
    sub_a_7d = serie_diaria.iloc[mot1_7d["idx_a"] : mot1_7d["idx_a"] + m7].to_numpy()
    sub_b_7d = serie_diaria.iloc[mot1_7d["idx_b"] : mot1_7d["idx_b"] + m7].to_numpy()
    d7 = np.arange(m7)
    axes[1, 1].plot(d7, _zscore_1d(sub_a_7d), "o-", color="#1f77b4", ms=5.5, lw=1.8, label=f"A: {mot1_7d['inicio_a'][:10]} (μ={mot1_7d['media_kw_a']:.1f} kW)")
    axes[1, 1].plot(d7, _zscore_1d(sub_b_7d), "s--", color="#d62728", ms=5.5, lw=1.6, label=f"B: {mot1_7d['inicio_b'][:10]} (μ={mot1_7d['media_kw_b']:.1f} kW)")
    axes[1, 1].set_title(f"Motif #1 Diário Agregado (m=7 dias) em Z-score | $d_Z$={mot1_7d['distancia_mp']:.2f}, r={mot1_7d['correlacao_pearson']:.4f}")
    axes[1, 1].set_xlabel("Dia relativo dentro da semana (0 a 6)")
    axes[1, 1].set_ylabel("Consumo padronizado (Z-score)")
    axes[1, 1].set_xticks(d7)
    axes[1, 1].legend(loc="lower right", fontsize=8.3)

    plt.tight_layout()
    p9 = dir_fig / "etapa2_motifs_sobrepostos.png"
    fig.savefig(p9, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p9))

    # ------------------------------------------------------------------
    # Figura 10 (etapa2_discords_contexto.png): Discords em contexto e forma
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    # Topo esquerdo: Discord #1 semanal em contexto (±14 dias)
    ctx_d = pd.Timedelta(days=14)
    ini_d1 = pd.Timestamp(disc1_168["inicio_discord"])
    fim_d1 = pd.Timestamp(disc1_168["fim_discord"])
    trecho_ctx1 = serie_horaria.loc[ini_d1 - ctx_d : fim_d1 + ctx_d]
    sub_d1 = serie_horaria.iloc[disc1_168["idx_discord"] : disc1_168["idx_discord"] + m168]
    axes[0, 0].plot(trecho_ctx1.index, trecho_ctx1.values, color="#1f77b4", lw=0.9, label="Contexto (±14 dias)")
    axes[0, 0].plot(sub_d1.index, sub_d1.values, color="#d62728", lw=1.4, label=f"Discord #1 ($d_Z$={disc1_168['distancia_mp']:.2f})")
    axes[0, 0].axvspan(ini_d1, fim_d1, color="#d62728", alpha=0.15)
    axes[0, 0].set_title(f"Discord #1 Semanal (m=168h) em Contexto: {str(ini_d1)[:10]} a {str(fim_d1)[:10]}")
    axes[0, 0].set_xlabel("Data")
    axes[0, 0].set_ylabel("Consumo (kW)")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].legend(loc="upper right", fontsize=8.3)

    # Topo direito: Discord #1 em Z-score vs vizinho mais próximo e semana mediana típica
    sub_v1 = serie_horaria.iloc[disc1_168["idx_vizinho"] : disc1_168["idx_vizinho"] + m168].to_numpy()
    # Alinha a semana mediana típica pelo dia da semana/hora da janela do discord
    idx_janela_d1 = serie_horaria.index[disc1_168["idx_discord"] : disc1_168["idx_discord"] + m168]
    horas_sem_d1 = idx_janela_d1.dayofweek * 24 + idx_janela_d1.hour
    mediana_168_alinhada = _perfil_semanal_mediano_z(serie_horaria)[horas_sem_d1]
    axes[0, 1].plot(h168, _zscore_1d(sub_d1.to_numpy()), color="#d62728", lw=1.6, label=f"Discord #1 ({str(ini_d1)[:10]})")
    axes[0, 1].plot(h168, _zscore_1d(sub_v1), color="#1f77b4", lw=1.3, ls="--", alpha=0.85, label=f"Vizinho mais próximo ({disc1_168['inicio_vizinho'][:10]}, r={disc1_168['correlacao_com_vizinho']:.3f})")
    axes[0, 1].plot(h168, mediana_168_alinhada, color="#2ca02c", lw=1.2, ls=":", label="Perfil semanal mediano da série")
    for dia_div in range(24, m168, 24):
        axes[0, 1].axvline(dia_div, color="#888888", lw=0.6, ls=":")
    axes[0, 1].set_title(f"Discord #1 vs. Vizinho Mais Próximo (Z-score) | $d_Z$={disc1_168['distancia_mp']:.2f}")
    axes[0, 1].set_xlabel("Horas dentro da janela (m=168h)")
    axes[0, 1].set_ylabel("Consumo padronizado (Z-score)")
    axes[0, 1].set_xticks(np.arange(0, m168 + 1, 24))
    axes[0, 1].legend(loc="lower left", fontsize=8.0)

    # Baixo esquerdo: Discord #2 semanal em contexto (±14 dias)
    ini_d2 = pd.Timestamp(disc2_168["inicio_discord"])
    fim_d2 = pd.Timestamp(disc2_168["fim_discord"])
    trecho_ctx2 = serie_horaria.loc[ini_d2 - ctx_d : fim_d2 + ctx_d]
    sub_d2 = serie_horaria.iloc[disc2_168["idx_discord"] : disc2_168["idx_discord"] + m168]
    axes[1, 0].plot(trecho_ctx2.index, trecho_ctx2.values, color="#1f77b4", lw=0.9, label="Contexto (±14 dias)")
    axes[1, 0].plot(sub_d2.index, sub_d2.values, color="#ff7f0e", lw=1.4, label=f"Discord #2 ($d_Z$={disc2_168['distancia_mp']:.2f})")
    axes[1, 0].axvspan(ini_d2, fim_d2, color="#ff7f0e", alpha=0.18)
    axes[1, 0].set_title(f"Discord #2 Semanal (m=168h) em Contexto: {str(ini_d2)[:10]} a {str(fim_d2)[:10]}")
    axes[1, 0].set_xlabel("Data")
    axes[1, 0].set_ylabel("Consumo (kW)")
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    axes[1, 0].tick_params(axis="x", rotation=20)
    axes[1, 0].legend(loc="upper right", fontsize=8.3)

    # Baixo direito: Discord #1 diário (m=24h) em contexto (±5 dias)
    ctx_5d = pd.Timedelta(days=5)
    ini_d24 = pd.Timestamp(disc1_24["inicio_discord"])
    fim_d24 = pd.Timestamp(disc1_24["fim_discord"])
    trecho_ctx24 = serie_horaria.loc[ini_d24 - ctx_5d : fim_d24 + ctx_5d]
    sub_d24 = serie_horaria.iloc[disc1_24["idx_discord"] : disc1_24["idx_discord"] + m24]
    axes[1, 1].plot(trecho_ctx24.index, trecho_ctx24.values, color="#1f77b4", lw=1.0, label="Contexto (±5 dias)")
    axes[1, 1].plot(sub_d24.index, sub_d24.values, color="#d62728", lw=1.7, label=f"Discord #1 24h ($d_Z$={disc1_24['distancia_mp']:.2f})")
    axes[1, 1].axvspan(ini_d24, fim_d24, color="#d62728", alpha=0.18)
    axes[1, 1].set_title(f"Discord #1 Intradiário (m=24h) em Contexto: {str(ini_d24)[:13]}h")
    axes[1, 1].set_xlabel("Data")
    axes[1, 1].set_ylabel("Consumo (kW)")
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(loc="upper right", fontsize=8.3)

    plt.tight_layout()
    p10 = dir_fig / "etapa2_discords_contexto.png"
    fig.savefig(p10, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p10))

    # ------------------------------------------------------------------
    # Figura 11 (etapa2_efeito_zscore_mp.png): Papel da normalização Z-score
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2))

    orig_info = exp_zscore["serie_original"]
    pert_info = exp_zscore["experimento_perturbacao"]
    mp_sem = orig_info["mp_sem_dist"]
    std_loc = orig_info["std_local"]

    # Topo esquerdo: MP com Z-score vs sem normalização (escalados por mediana para comparar variação)
    axes[0, 0].plot(idx_mp168, mp_sem / np.median(mp_sem), color="#d62728", lw=0.7, alpha=0.8, label="Sem normalização (dividido pela mediana)")
    axes[0, 0].plot(idx_mp168, mp168_dist / np.median(mp168_dist), color="#1f77b4", lw=0.7, alpha=0.85, label="Com Z-score (dividido pela mediana)")
    disc_sem_1 = orig_info["top3_discords_sem_norm"][0]
    axes[0, 0].scatter(
        [pd.Timestamp(disc_sem_1["inicio_discord"])],
        [disc_sem_1["distancia_mp"] / np.median(mp_sem)],
        color="#900000", s=55, marker="X", zorder=5,
        label=f"Pico sem Z-score ({disc_sem_1['inicio_discord'][:10]})",
    )
    axes[0, 0].set_title("MP Relativo (m=168h): Sem Normalização vs. Z-score na Série Original")
    axes[0, 0].set_xlabel("Data (2011–2014)")
    axes[0, 0].set_ylabel("Distância MP relativa à mediana")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(loc="upper right", fontsize=8.0)

    # Topo direito: Correlação espúria com desvio-padrão local quando não há Z-score
    passo_amostra = 6  # subamostra visual uniforme para legibilidade do scatter
    axes[0, 1].scatter(std_loc[::passo_amostra], mp_sem[::passo_amostra], s=7, alpha=0.35, color="#d62728", label=f"Sem Z-score (r com σ_local = {orig_info['corr_mp_sem_norm_com_desvio_local']:.3f})")
    ax_twin = axes[0, 1].twinx()
    ax_twin.scatter(std_loc[::passo_amostra], mp168_dist[::passo_amostra], s=7, alpha=0.30, color="#1f77b4", label=f"Com Z-score (r com σ_local = {orig_info['corr_mp_zscore_com_desvio_local']:.3f})")
    ax_twin.grid(False)
    axes[0, 1].set_title("Sensibilidade à Amplitude Local (σ da janela de 168h)")
    axes[0, 1].set_xlabel("Desvio-padrão local da subsequência σ (kW)")
    axes[0, 1].set_ylabel("Distância MP sem normalização (kW)", color="#d62728")
    ax_twin.set_ylabel("Distância MP Z-score ($d_Z$)", color="#1f77b4")
    h1, l1 = axes[0, 1].get_legend_handles_labels()
    h2, l2 = ax_twin.get_legend_handles_labels()
    axes[0, 1].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.0)

    # Baixo esquerdo: Experimento controlado na janela alvo (kW perturbado vs Z-score invariante)
    idx_alvo = pert_info["idx_alvo"]
    sub_orig_alvo = serie_horaria.iloc[idx_alvo : idx_alvo + m168].to_numpy()
    sub_pert_alvo = pert_info["serie_pert"].iloc[idx_alvo : idx_alvo + m168].to_numpy()
    axes[1, 0].plot(h168, sub_orig_alvo, color="#1f77b4", lw=1.6, label="Original (Motif #1)")
    axes[1, 0].plot(h168, sub_pert_alvo, color="#d62728", lw=1.4, ls="--", label=f"Escalada ({pert_info['fator_escala']}x + {pert_info['offset_kw']:.0f} kW)")
    ax_z_inset = axes[1, 0].twinx()
    ax_z_inset.plot(h168, _zscore_1d(sub_orig_alvo), color="#2ca02c", lw=1.2, alpha=0.75, label="Z-score original = Z-score escalado")
    ax_z_inset.grid(False)
    axes[1, 0].set_title(f"Experimento Controlado na Janela {pert_info['inicio_alvo'][:10]} (Forma Idêntica)")
    axes[1, 0].set_xlabel("Horas dentro da janela (m=168h)")
    axes[1, 0].set_ylabel("Potência (kW)")
    ax_z_inset.set_ylabel("Forma Z-score (sobreposta)", color="#2ca02c")
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    h2, l2 = ax_z_inset.get_legend_handles_labels()
    axes[1, 0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.8)

    # Baixo direito: Comparação ao redor da janela perturbada no MP com e sem Z-score
    margem = 24 * 14
    s_i = max(0, idx_alvo - margem)
    s_f = min(len(mp168_dist), idx_alvo + m168 + margem)
    idx_recorte = serie_horaria.index[s_i:s_f]
    axes[1, 1].plot(
        idx_recorte,
        pert_info["pert_sem_dist"][s_i:s_f] / np.median(pert_info["pert_sem_dist"]),
        color="#d62728",
        lw=1.3,
        label=f"Sem Z-score: salta {pert_info['razao_salto_sem_norm']:.1f}x (rank discord #{pert_info['rank_discord_pert_sem_norm']})",
    )
    axes[1, 1].plot(
        idx_recorte,
        pert_info["pert_norm_dist"][s_i:s_f] / np.median(pert_info["pert_norm_dist"]),
        color="#1f77b4",
        lw=1.4,
        label=f"Com Z-score: Δ={pert_info['delta_abs_zscore']:.4f} (mantém Motif #{pert_info['rank_motif_pert_zscore']})",
    )
    axes[1, 1].axvline(pd.Timestamp(pert_info["inicio_alvo"]), color="#222222", ls=":", lw=1.2, label="Início da janela perturbada")
    axes[1, 1].set_title("Impacto da Perturbação Afim no Matrix Profile (Relativo à Mediana)")
    axes[1, 1].set_xlabel("Data")
    axes[1, 1].set_ylabel("Distância MP relativa à mediana")
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%y"))
    axes[1, 1].tick_params(axis="x", rotation=20)
    axes[1, 1].legend(loc="upper right", fontsize=7.8)

    plt.tight_layout()
    p11 = dir_fig / "etapa2_efeito_zscore_mp.png"
    fig.savefig(p11, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p11))

    return arquivos_salvos


def salvar_saidas_matrix_profile(
    cliente: str,
    resumo_serie: dict[str, Any],
    estatisticas_mp: dict[str, dict[str, Any]],
    motifs_por_janela: dict[str, list[dict[str, Any]]],
    discords_por_janela: dict[str, list[dict[str, Any]]],
    exp_zscore: dict[str, Any],
    figuras: list[str],
    dir_resultados: Path | str = DIRETORIO_RESULTADOS_PADRAO,
) -> dict[str, str]:
    """Persiste métricas estruturadas em JSON e tabela resumo de motifs/discords em CSV."""
    p_res = Path(dir_resultados)
    p_res.mkdir(parents=True, exist_ok=True)

    # Remove arrays grandes do dicionário antes de serializar em JSON
    exp_serializavel = {
        "serie_original": {
            k: v
            for k, v in exp_zscore["serie_original"].items()
            if k not in {"mp_sem_dist", "std_local"}
        },
        "experimento_perturbacao": {
            k: v
            for k, v in exp_zscore["experimento_perturbacao"].items()
            if k not in {"serie_pert", "pert_norm_dist", "pert_sem_dist"}
        },
    }

    payload = {
        "semente": SEMENTE_PADRAO,
        "cliente_representativo": cliente,
        "resumo_serie": resumo_serie,
        "estatisticas_mp": estatisticas_mp,
        "motifs": motifs_por_janela,
        "discords": discords_por_janela,
        "experimento_zscore": exp_serializavel,
        "figuras_geradas": figuras,
    }

    arq_json = p_res / "metricas_matrix_profile.json"
    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    linhas_csv: list[dict[str, Any]] = []
    for escala, lista in motifs_por_janela.items():
        for item in lista:
            linhas_csv.append(
                {
                    "escala_janela": escala,
                    "tipo": "motif",
                    "rank": item["rank"],
                    "m": item["m"],
                    "inicio_principal": item["inicio_a"],
                    "fim_principal": item["fim_a"],
                    "inicio_par_ou_vizinho": item["inicio_b"],
                    "fim_par_ou_vizinho": item["fim_b"],
                    "distancia_mp": item["distancia_mp"],
                    "correlacao_pearson": item["correlacao_pearson"],
                    "media_kw_principal": item["media_kw_a"],
                    "media_kw_par_ou_vizinho": item["media_kw_b"],
                }
            )
    for escala, lista in discords_por_janela.items():
        for item in lista:
            linhas_csv.append(
                {
                    "escala_janela": escala,
                    "tipo": "discord",
                    "rank": item["rank"],
                    "m": item["m"],
                    "inicio_principal": item["inicio_discord"],
                    "fim_principal": item["fim_discord"],
                    "inicio_par_ou_vizinho": item["inicio_vizinho"],
                    "fim_par_ou_vizinho": item["fim_vizinho"],
                    "distancia_mp": item["distancia_mp"],
                    "correlacao_pearson": item["correlacao_com_vizinho"],
                    "media_kw_principal": item["media_kw_discord"],
                    "media_kw_par_ou_vizinho": item["media_kw_vizinho"],
                }
            )

    arq_csv = p_res / "resumo_motifs_discords_etapa2.csv"
    pd.DataFrame(linhas_csv).to_csv(arq_csv, index=False)

    return {
        "metricas_json": str(arq_json),
        "resumo_csv": str(arq_csv),
    }
