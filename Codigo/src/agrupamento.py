"""Módulo de agrupamento de séries temporais (Etapa 1): DTW + AGNES + Coeficiente de Silhueta."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples, silhouette_score
from tslearn.metrics import cdist_dtw

# Semente padrão e hiperparâmetros explícitos (sem números mágicos espalhados)
SEMENTE_PADRAO: int = 42
RAIO_SAKOE_CHIBA_DIAS: int = 7
INTERVALO_K: tuple[int, ...] = tuple(range(2, 11))
METODOS_LIGACAO: tuple[str, ...] = ("average", "ward")

# Caminhos padrão relativos à raiz do repositório
CAMINHO_DIARIO_PADRAO: Path = Path("Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet")
CAMINHO_HORARIO_PADRAO: Path = Path("Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet")
CAMINHO_CACHE_DTW: Path = Path("Resultados/matriz_dtw_diario_154.npy")
DIRETORIO_RESULTADOS_PADRAO: Path = Path("Resultados")
DIRETORIO_FIGURAS_PADRAO: Path = Path("Documentacao/figuras")


def normalizar_zscore_por_serie(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza cada série temporal (coluna) para média zero e desvio-padrão unitário.

    Garante que a distância DTW compare a forma e a dinâmica temporal
    (tendência e sazonalidade), evitando que diferenças de magnitude em kW
    dominem o agrupamento.
    """
    medias = df.mean(axis=0)
    desvios = df.std(axis=0, ddof=0).replace(0.0, 1.0)
    return (df - medias) / desvios


def calcular_ou_carregar_matriz_dtw(
    df_z: pd.DataFrame,
    raio: int = RAIO_SAKOE_CHIBA_DIAS,
    caminho_cache: Path | str = CAMINHO_CACHE_DTW,
    forcar_recalculo: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Calcula ou carrega em cache a matriz simétrica de distâncias L2-DTW com Sakoe-Chiba.

    Usa `tslearn.metrics.cdist_dtw` com restrição global de Sakoe-Chiba (`raio` dias)
    sobre as séries transpostas `(n_clientes, n_passos)`.
    """
    caminho = Path(caminho_cache)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    n_series, n_passos = df_z.shape[1], df_z.shape[0]
    carregado_do_cache = False
    tempo_calculo_seg = 0.0

    if caminho.exists() and not forcar_recalculo:
        matriz = np.load(caminho)
        if matriz.shape == (n_series, n_series):
            carregado_do_cache = True
        else:
            matriz = None
    else:
        matriz = None

    if matriz is None:
        x = df_z.to_numpy(dtype=np.float64).T
        t0 = time.perf_counter()
        matriz = cdist_dtw(
            x,
            global_constraint="sakoe_chiba",
            sakoe_chiba_radius=raio,
            n_jobs=-1,
        )
        tempo_calculo_seg = float(time.perf_counter() - t0)
        # Garante simetria numérica estrita e diagonal exatamente zero para o squareform
        matriz = 0.5 * (matriz + matriz.T)
        np.fill_diagonal(matriz, 0.0)
        np.save(caminho, matriz)

    tri_sup = matriz[np.triu_indices(n_series, k=1)]
    estatisticas = {
        "n_series": int(n_series),
        "comprimento_serie": int(n_passos),
        "n_pares_unicos": int(len(tri_sup)),
        "restricao_global": "sakoe_chiba",
        "sakoe_chiba_radius_dias": int(raio),
        "carregado_do_cache": carregado_do_cache,
        "tempo_calculo_seg": round(tempo_calculo_seg, 3),
        "caminho_cache": str(caminho),
        "distancia_min_nao_nula": round(float(tri_sup.min()), 4),
        "distancia_mediana": round(float(np.median(tri_sup)), 4),
        "distancia_media": round(float(tri_sup.mean()), 4),
        "distancia_max": round(float(tri_sup.max()), 4),
    }
    return matriz, estatisticas


def _identificar_medoide(
    matriz_dtw: np.ndarray,
    indices_cluster: np.ndarray,
    nomes_clientes: list[str],
) -> str:
    """Retorna o nome da série medóide (menor distância DTW média para as demais do cluster)."""
    if len(indices_cluster) == 1:
        return nomes_clientes[int(indices_cluster[0])]
    sub = matriz_dtw[np.ix_(indices_cluster, indices_cluster)]
    pos_local = int(np.argmin(sub.mean(axis=1)))
    return nomes_clientes[int(indices_cluster[pos_local])]


def _altura_corte_para_k(Z: np.ndarray, k: int) -> float:
    """Calcula a altura horizontal de corte no dendrograma que gera exatamente `k` clusters."""
    # As últimas k-1 fusões unem os k clusters; o corte fica no ponto médio entre a (k-1)-ésima e a k-ésima
    alto = float(Z[-(k - 1), 2])
    baixo = float(Z[-k, 2])
    return 0.5 * (alto + baixo)


def avaliar_agnes_silhueta(
    matriz_dtw: np.ndarray,
    nomes_clientes: list[str],
    metodos: tuple[str, ...] = METODOS_LIGACAO,
    ks: tuple[int, ...] = INTERVALO_K,
) -> dict[str, Any]:
    """Executa AGNES (`average` e `ward`) sobre a matriz DTW e avalia a Silhueta para vários k."""
    dist_condensada = squareform(matriz_dtw, checks=True)
    resultados: dict[str, Any] = {}

    for metodo in metodos:
        Z = linkage(dist_condensada, method=metodo)
        avaliacoes_k: dict[int, dict[str, Any]] = {}
        rotulos_por_k: dict[int, np.ndarray] = {}
        silhuetas_por_k: dict[int, np.ndarray] = {}

        for k in ks:
            rotulos = fcluster(Z, k, criterion="maxclust")
            sil_media = float(silhouette_score(matriz_dtw, rotulos, metric="precomputed"))
            sil_amostras = silhouette_samples(matriz_dtw, rotulos, metric="precomputed")

            clusters_info: list[dict[str, Any]] = []
            for c_id in sorted(np.unique(rotulos).tolist()):
                idx_c = np.where(rotulos == c_id)[0]
                medoide = _identificar_medoide(matriz_dtw, idx_c, nomes_clientes)
                clusters_info.append(
                    {
                        "cluster": int(c_id),
                        "tamanho": int(len(idx_c)),
                        "silhueta_media": round(float(sil_amostras[idx_c].mean()), 4),
                        "medoide": medoide,
                    }
                )

            negativos = [
                {
                    "cliente": nomes_clientes[i],
                    "cluster": int(rotulos[i]),
                    "silhueta": round(float(sil_amostras[i]), 4),
                }
                for i in range(len(nomes_clientes))
                if sil_amostras[i] < 0.0
            ]
            negativos.sort(key=lambda item: item["silhueta"])

            avaliacoes_k[k] = {
                "k": int(k),
                "altura_corte_dendrograma": round(_altura_corte_para_k(Z, k), 4),
                "silhueta_media": round(sil_media, 4),
                "n_negativos": len(negativos),
                "tamanhos_clusters": [c["tamanho"] for c in clusters_info],
                "clusters": clusters_info,
                "series_negativas": negativos,
            }
            rotulos_por_k[k] = rotulos
            silhuetas_por_k[k] = sil_amostras

        resultados[metodo] = {
            "linkage_matrix": Z,
            "avaliacoes_k": avaliacoes_k,
            "rotulos_por_k": rotulos_por_k,
            "silhuetas_por_k": silhuetas_por_k,
        }

    return resultados


def resumir_perfis_clusters(
    df_diario: pd.DataFrame,
    df_horario: pd.DataFrame,
    matriz_dtw: np.ndarray,
    rotulos: np.ndarray,
    sil_amostras: np.ndarray,
    nome_particao: str,
) -> pd.DataFrame:
    """Gera resumo físico e estatístico de cada cluster em uma partição selecionada."""
    nomes_clientes = list(df_diario.columns)
    linhas: list[dict[str, Any]] = []

    mask_2011 = df_diario.index.year == 2011
    mask_2014 = df_diario.index.year == 2014
    mask_verao = df_diario.index.month.isin([6, 7, 8])
    mask_inverno = df_diario.index.month.isin([12, 1, 2])
    mask_util = df_horario.index.dayofweek < 5
    mask_fds = df_horario.index.dayofweek >= 5

    for c_id in sorted(np.unique(rotulos).tolist()):
        idx_c = np.where(rotulos == c_id)[0]
        cols_c = [nomes_clientes[i] for i in idx_c]
        sub_d = df_diario[cols_c]
        sub_h = df_horario[cols_c]

        medoide = _identificar_medoide(matriz_dtw, idx_c, nomes_clientes)
        consumo_medio_por_cliente = sub_d.mean(axis=0)

        m_2011 = float(sub_d.loc[mask_2011].mean().mean())
        m_2014 = float(sub_d.loc[mask_2014].mean().mean())
        m_verao = float(sub_d.loc[mask_verao].mean().mean())
        m_inverno = float(sub_d.loc[mask_inverno].mean().mean())
        m_util = float(sub_h.loc[mask_util].mean().mean())
        m_fds = float(sub_h.loc[mask_fds].mean().mean())

        # Razões individuais por cliente (mediana não ponderada por carga em kW)
        razao_2014_2011_cli = sub_d.loc[mask_2014].mean(axis=0) / sub_d.loc[mask_2011].mean(axis=0)
        razao_verao_inverno_cli = sub_d.loc[mask_verao].mean(axis=0) / sub_d.loc[mask_inverno].mean(axis=0)

        # Horário típico de pico a partir do perfil horário médio padronizado do grupo
        perfil_24h = sub_h.groupby(sub_h.index.hour).mean()
        perfil_24h_z = ((perfil_24h - perfil_24h.mean()) / (perfil_24h.std() + 1e-9)).mean(axis=1)
        hora_pico = int(perfil_24h_z.idxmax())

        linhas.append(
            {
                "particao": nome_particao,
                "cluster": int(c_id),
                "n_clientes": int(len(cols_c)),
                "pct_clientes": round(100.0 * len(cols_c) / len(nomes_clientes), 2),
                "medoide": medoide,
                "silhueta_media": round(float(sil_amostras[idx_c].mean()), 4),
                "silhueta_min": round(float(sil_amostras[idx_c].min()), 4),
                "silhueta_max": round(float(sil_amostras[idx_c].max()), 4),
                "consumo_medio_kw": round(float(consumo_medio_por_cliente.mean()), 2),
                "consumo_mediano_kw": round(float(consumo_medio_por_cliente.median()), 2),
                "razao_tendencia_2014_2011": round(m_2014 / m_2011, 3),
                "mediana_tendencia_2014_2011": round(float(razao_2014_2011_cli.median()), 3),
                "razao_verao_inverno": round(m_verao / m_inverno, 3),
                "mediana_verao_inverno": round(float(razao_verao_inverno_cli.median()), 3),
                "razao_dia_util_fim_semana": round(m_util / m_fds, 3),
                "hora_pico_media": hora_pico,
                "exemplos_clientes": ", ".join(cols_c[:6]),
            }
        )

    return pd.DataFrame(linhas)


def gerar_tabela_clientes_agrupamento(
    df_diario: pd.DataFrame,
    resultados_agnes: dict[str, Any],
) -> pd.DataFrame:
    """Consolida rótulos e silhuetas individuais de todos os 154 clientes nas partições-chave."""
    nomes = list(df_diario.columns)
    mask_2011 = df_diario.index.year == 2011
    mask_2014 = df_diario.index.year == 2014
    mask_verao = df_diario.index.month.isin([6, 7, 8])
    mask_inverno = df_diario.index.month.isin([12, 1, 2])

    df_out = pd.DataFrame(
        {
            "cliente": nomes,
            "consumo_medio_kw": df_diario.mean(axis=0).round(2).values,
            "razao_2014_2011": (df_diario.loc[mask_2014].mean() / df_diario.loc[mask_2011].mean()).round(3).values,
            "razao_verao_inverno": (df_diario.loc[mask_verao].mean() / df_diario.loc[mask_inverno].mean()).round(3).values,
        }
    )

    configuracoes = [
        ("average", 2),
        ("average", 3),
        ("average", 5),
        ("ward", 2),
        ("ward", 3),
        ("ward", 4),
    ]
    for metodo, k in configuracoes:
        df_out[f"cluster_{metodo}_k{k}"] = resultados_agnes[metodo]["rotulos_por_k"][k]
        df_out[f"sil_{metodo}_k{k}"] = np.round(resultados_agnes[metodo]["silhuetas_por_k"][k], 4)

    return df_out


def _plotar_silhueta_painel(
    ax: plt.Axes,
    sil_amostras: np.ndarray,
    rotulos: np.ndarray,
    nomes_clientes: list[str],
    titulo: str,
    anotar_negativos: bool = False,
) -> None:
    """Desenha um gráfico detalhado de silhueta por cluster e por amostra em um Axes."""
    sil_media = float(sil_amostras.mean())
    n_neg = int((sil_amostras < 0).sum())
    clusters_ordenados = sorted(np.unique(rotulos).tolist())
    k = len(clusters_ordenados)

    y_inferior = 8
    paleta = cm.get_cmap("tab10", max(k, 3))

    posicoes_negativas: list[tuple[float, float, str]] = []
    for idx_cor, c_id in enumerate(clusters_ordenados):
        vals = sil_amostras[rotulos == c_id]
        nomes_c = [nomes_clientes[i] for i in np.where(rotulos == c_id)[0]]
        ordem = np.argsort(vals)
        vals_ord = vals[ordem]
        nomes_ord = [nomes_c[i] for i in ordem]

        tam = len(vals_ord)
        y_superior = y_inferior + tam
        cor = paleta(idx_cor)

        ax.fill_betweenx(
            np.arange(y_inferior, y_superior),
            0,
            vals_ord,
            facecolor=cor,
            edgecolor=cor,
            alpha=0.82,
        )
        # Rótulo do cluster na margem esquerda fixa, livre da região de barras negativas
        y_centro = y_inferior + 0.5 * tam
        ax.text(
            -0.46,
            y_centro,
            f"C{c_id} (n={tam})",
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold",
            color="#222222",
        )

        if anotar_negativos:
            for pos_local, (sv, nm) in enumerate(zip(vals_ord, nomes_ord)):
                if sv < 0.0:
                    posicoes_negativas.append((float(sv), float(y_inferior + pos_local), nm))

        y_inferior = y_superior + 10

    ax.axvline(x=0.0, color="#555555", linestyle="-", lw=0.8)
    ax.axvline(
        x=sil_media,
        color="#d62728",
        linestyle="--",
        lw=1.6,
        label=f"Média global = {sil_media:.4f} (neg={n_neg})",
    )

    if anotar_negativos and len(posicoes_negativas) <= 5:
        deslocamentos = [12, -10, 14, -12, 16]
        for idx_neg, (sv, y_pos, nm) in enumerate(posicoes_negativas):
            desloc_y = deslocamentos[idx_neg % len(deslocamentos)]
            ax.annotate(
                f"{nm} ({sv:.3f})",
                xy=(sv, y_pos),
                xytext=(-0.31, y_pos + desloc_y),
                ha="left",
                fontsize=7.5,
                color="#900000",
                arrowprops={"arrowstyle": "->", "color": "#900000", "lw": 0.8},
            )

    ax.set_title(titulo, fontsize=10.5)
    ax.set_xlabel("Coeficiente de Silhueta ($s_i$)")
    ax.set_ylabel("Amostras agrupadas por cluster")
    ax.set_xlim(-0.48, 0.85)
    ax.set_ylim(-6, y_inferior + 4)
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=8.5)


def gerar_figuras_agrupamento(
    df_diario: pd.DataFrame,
    df_horario: pd.DataFrame,
    resultados_agnes: dict[str, Any],
    diretorio_figuras: Path | str = DIRETORIO_FIGURAS_PADRAO,
) -> list[str]:
    """Gera e salva as 4 figuras principais da Etapa 1 em Documentacao/figuras/."""
    dir_fig = Path(diretorio_figuras)
    dir_fig.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3})

    nomes_clientes = list(df_diario.columns)
    arquivos_salvos: list[str] = []

    # ------------------------------------------------------------------
    # Figura 4 (etapa1_dendrograma.png): Dendrogramas Average vs Ward
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    # Painel A: Average linkage
    Z_avg = resultados_agnes["average"]["linkage_matrix"]
    av_avg_k2 = resultados_agnes["average"]["avaliacoes_k"][2]
    corte_avg_k2 = av_avg_k2["altura_corte_dendrograma"]
    sil_avg_k2 = av_avg_k2["silhueta_media"]
    dendrogram(
        Z_avg,
        ax=axes[0],
        truncate_mode="lastp",
        p=35,
        show_contracted=True,
        leaf_rotation=90,
        leaf_font_size=7.5,
        color_threshold=corte_avg_k2,
        above_threshold_color="#444444",
    )
    axes[0].axhline(
        corte_avg_k2,
        color="#d62728",
        linestyle="--",
        lw=1.7,
        label=f"Corte ótimo k=2 (h={corte_avg_k2:.2f}, sil={sil_avg_k2:.4f})",
    )
    axes[0].set_title("Dendrograma AGNES — Ligação Average (DTW, r=7 dias)")
    axes[0].set_xlabel("Ramos agrupados (número entre parênteses = tamanho do ramo)")
    axes[0].set_ylabel("Distância DTW inter-cluster")
    axes[0].legend(loc="upper right", fontsize=9)

    # Painel B: Ward linkage
    Z_ward = resultados_agnes["ward"]["linkage_matrix"]
    av_ward_k2 = resultados_agnes["ward"]["avaliacoes_k"][2]
    av_ward_k3 = resultados_agnes["ward"]["avaliacoes_k"][3]
    corte_ward_k2 = av_ward_k2["altura_corte_dendrograma"]
    corte_ward_k3 = av_ward_k3["altura_corte_dendrograma"]
    sil_ward_k2 = av_ward_k2["silhueta_media"]
    sil_ward_k3 = av_ward_k3["silhueta_media"]
    dendrogram(
        Z_ward,
        ax=axes[1],
        truncate_mode="lastp",
        p=35,
        show_contracted=True,
        leaf_rotation=90,
        leaf_font_size=7.5,
        color_threshold=corte_ward_k3,
        above_threshold_color="#444444",
    )
    axes[1].axhline(
        corte_ward_k2,
        color="#d62728",
        linestyle="--",
        lw=1.7,
        label=f"Corte k=2 (h={corte_ward_k2:.2f}, sil={sil_ward_k2:.4f})",
    )
    axes[1].axhline(
        corte_ward_k3,
        color="#ff7f0e",
        linestyle="-.",
        lw=1.7,
        label=f"Corte secundário k=3 (h={corte_ward_k3:.2f}, sil={sil_ward_k3:.4f})",
    )
    axes[1].set_title("Dendrograma AGNES — Ligação Ward (DTW, r=7 dias)")
    axes[1].set_xlabel("Ramos agrupados (número entre parênteses = tamanho do ramo)")
    axes[1].set_ylabel("Dissimilaridade Ward sobre matriz DTW")
    axes[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    p_dend = dir_fig / "etapa1_dendrograma.png"
    fig.savefig(p_dend, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p_dend))

    # ------------------------------------------------------------------
    # Figura 5 (etapa1_silhueta_por_k.png): Curvas Silhueta média e Negativos x k
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ks = sorted(resultados_agnes["average"]["avaliacoes_k"].keys())

    sil_avg = [resultados_agnes["average"]["avaliacoes_k"][k]["silhueta_media"] for k in ks]
    sil_ward = [resultados_agnes["ward"]["avaliacoes_k"][k]["silhueta_media"] for k in ks]
    neg_avg = [resultados_agnes["average"]["avaliacoes_k"][k]["n_negativos"] for k in ks]
    neg_ward = [resultados_agnes["ward"]["avaliacoes_k"][k]["n_negativos"] for k in ks]

    axes[0].plot(ks, sil_avg, "o-", color="#1f77b4", lw=2.0, ms=6, label="AGNES Average")
    axes[0].plot(ks, sil_ward, "s--", color="#d62728", lw=2.0, ms=6, label="AGNES Ward")
    axes[0].axvline(2, color="#2ca02c", linestyle=":", lw=1.6, label="Corte ótimo global (k=2)")
    axes[0].annotate(
        f"Máximo: {sil_avg[0]:.4f}\n(k=2, ambos)",
        xy=(2, sil_avg[0]),
        xytext=(3.1, sil_avg[0] + 0.015),
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 0.9},
    )
    axes[0].annotate(
        f"Ward k=3: {sil_ward[1]:.4f}",
        xy=(3, sil_ward[1]),
        xytext=(4.0, sil_ward[1] + 0.025),
        fontsize=8.5,
        color="#a01010",
        arrowprops={"arrowstyle": "->", "color": "#a01010", "lw": 0.9},
    )
    axes[0].set_title("Coeficiente de Silhueta Médio × Número de Clusters (k)")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Silhueta média (metric='precomputed')")
    axes[0].set_xticks(ks)
    axes[0].set_ylim(0.0, 0.56)
    axes[0].legend(loc="upper right")

    axes[1].plot(ks, neg_avg, "o-", color="#1f77b4", lw=2.0, ms=6, label="AGNES Average")
    axes[1].plot(ks, neg_ward, "s--", color="#d62728", lw=2.0, ms=6, label="AGNES Ward")
    axes[1].axvline(2, color="#2ca02c", linestyle=":", lw=1.6, label="Zero negativos (k=2)")
    axes[1].set_title("Séries com Silhueta Negativa × Número de Clusters (k)")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Quantidade de clientes com silhueta < 0 (de 154)")
    axes[1].set_xticks(ks)
    axes[1].set_ylim(-2, max(max(neg_ward), 10) + 5)
    axes[1].legend(loc="upper left")

    plt.tight_layout()
    p_sil_k = dir_fig / "etapa1_silhueta_por_k.png"
    fig.savefig(p_sil_k, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p_sil_k))

    # ------------------------------------------------------------------
    # Figura 6 (etapa1_silhueta_detalhada.png): Silhueta detalhada (2x2)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.8))
    n_neg_avg5 = resultados_agnes["average"]["avaliacoes_k"][5]["n_negativos"]
    n_neg_ward3 = resultados_agnes["ward"]["avaliacoes_k"][3]["n_negativos"]
    n_neg_ward4 = resultados_agnes["ward"]["avaliacoes_k"][4]["n_negativos"]
    paineis = [
        ("average", 2, "(a) Average / Ward — k=2 (Corte ótimo global)", False),
        ("average", 5, f"(b) Average — k=5 (Isolamento de singletons + {n_neg_avg5} negativos)", True),
        ("ward", 3, f"(c) Ward — k=3 (Subgrupo sazonal separado + {n_neg_ward3} negativos)", True),
        ("ward", 4, f"(d) Ward — k=4 (Degradação acentuada: {n_neg_ward4} negativos)", False),
    ]
    for ax, (metodo, k, titulo, anotar) in zip(axes.flat, paineis):
        _plotar_silhueta_painel(
            ax=ax,
            sil_amostras=resultados_agnes[metodo]["silhuetas_por_k"][k],
            rotulos=resultados_agnes[metodo]["rotulos_por_k"][k],
            nomes_clientes=nomes_clientes,
            titulo=titulo,
            anotar_negativos=anotar,
        )

    plt.tight_layout()
    p_sil_det = dir_fig / "etapa1_silhueta_detalhada.png"
    fig.savefig(p_sil_det, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p_sil_det))

    # ------------------------------------------------------------------
    # Figura 7 (etapa1_perfis_clusters.png): Interpretação física dos clusters
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0))
    df_z_d = normalizar_zscore_por_serie(df_diario)
    df_z_d_mm30 = df_z_d.rolling(30, center=True).mean()

    # Painel A: Evolução plurianual Z-score para k=2 (Average/Ward)
    rot_k2 = resultados_agnes["average"]["rotulos_por_k"][2]
    cores_k2 = {1: "#1f77b4", 2: "#d62728"}
    for c_id in sorted(np.unique(rot_k2).tolist()):
        cols_c = df_diario.columns[rot_k2 == c_id]
        info_c = next(
            c for c in resultados_agnes["average"]["avaliacoes_k"][2]["clusters"] if c["cluster"] == c_id
        )
        medoide = info_c["medoide"]
        curva_media = df_z_d_mm30[cols_c].mean(axis=1)
        p25 = df_z_d_mm30[cols_c].quantile(0.25, axis=1)
        p75 = df_z_d_mm30[cols_c].quantile(0.75, axis=1)
        cor = cores_k2.get(c_id, "#2ca02c")
        axes[0, 0].plot(
            curva_media.index,
            curva_media.values,
            color=cor,
            lw=2.0,
            label=f"Cluster {c_id} (n={len(cols_c)}, medóide={medoide})",
        )
        axes[0, 0].fill_between(curva_media.index, p25.values, p75.values, color=cor, alpha=0.16)
    axes[0, 0].set_title("Dinâmica plurianual Z-score (média móvel 30d) — Corte ótimo k=2")
    axes[0, 0].set_xlabel("Data (2011–2014)")
    axes[0, 0].set_ylabel("Consumo diário padronizado (Z-score)")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(loc="upper right", fontsize=8.5)

    # Painel B: Evolução plurianual Z-score para k=3 (Ward) — rótulos inferidos dos dados
    rot_w3 = resultados_agnes["ward"]["rotulos_por_k"][3]
    ids_w3 = sorted(np.unique(rot_w3).tolist())
    # Classifica dinamicamente os clusters de ward k=3 pelas propriedades físicas (2014/2011 e Verão/Inverno)
    metricas_w3 = {}
    for c_id in ids_w3:
        cols_c = df_diario.columns[rot_w3 == c_id]
        r_tend = float(df_diario.loc[df_diario.index.year == 2014, cols_c].mean().mean() / df_diario.loc[df_diario.index.year == 2011, cols_c].mean().mean())
        r_vi = float(df_diario.loc[df_diario.index.month.isin([6, 7, 8]), cols_c].mean().mean() / df_diario.loc[df_diario.index.month.isin([12, 1, 2]), cols_c].mean().mean())
        metricas_w3[c_id] = {"tend": r_tend, "vi": r_vi}
    id_expansao = max(ids_w3, key=lambda c: metricas_w3[c]["tend"])
    restantes = [c for c in ids_w3 if c != id_expansao]
    id_inverno = min(restantes, key=lambda c: metricas_w3[c]["vi"])
    id_verao = max(restantes, key=lambda c: metricas_w3[c]["vi"])
    cores_w3 = {id_expansao: "#d62728", id_inverno: "#2ca02c", id_verao: "#1f77b4"}
    nomes_desc_w3 = {
        id_expansao: "Expansão plurianual",
        id_inverno: "Inverno dominante",
        id_verao: "Verão dominante",
    }
    for c_id in ids_w3:
        cols_c = df_diario.columns[rot_w3 == c_id]
        info_c = next(
            c for c in resultados_agnes["ward"]["avaliacoes_k"][3]["clusters"] if c["cluster"] == c_id
        )
        medoide = info_c["medoide"]
        curva_media = df_z_d_mm30[cols_c].mean(axis=1)
        cor = cores_w3.get(c_id, "#9467bd")
        axes[0, 1].plot(
            curva_media.index,
            curva_media.values,
            color=cor,
            lw=2.0,
            label=f"C{c_id} {nomes_desc_w3.get(c_id, '')} (n={len(cols_c)}, med={medoide})",
        )
    axes[0, 1].set_title("Dinâmica plurianual Z-score (média móvel 30d) — Ward k=3")
    axes[0, 1].set_xlabel("Data (2011–2014)")
    axes[0, 1].set_ylabel("Consumo diário padronizado (Z-score)")
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 1].legend(loc="upper right", fontsize=8.5)

    # Painel C: Sazonalidade mensal média (Jan–Dez) dos 3 grupos + série negativa de transição
    meses = np.arange(1, 13)
    nomes_meses = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    saz_mensal = df_z_d.groupby(df_z_d.index.month).mean()
    for c_id in ids_w3:
        cols_c = df_diario.columns[rot_w3 == c_id]
        perfil_m = saz_mensal[cols_c].mean(axis=1)
        axes[1, 0].plot(
            meses,
            perfil_m.values,
            "o-",
            color=cores_w3.get(c_id, "#9467bd"),
            lw=2.0,
            ms=5,
            label=f"C{c_id} {nomes_desc_w3.get(c_id, '')} (n={len(cols_c)})",
        )
    # Destaca dinamicamente a série com menor silhueta negativa em ward k=3
    neg_w3 = resultados_agnes["ward"]["avaliacoes_k"][3]["series_negativas"]
    if neg_w3:
        neg_top = neg_w3[0]
        cli_neg = neg_top["cliente"]
        sil_neg = neg_top["silhueta"]
        if cli_neg in saz_mensal.columns:
            axes[1, 0].plot(
                meses,
                saz_mensal[cli_neg].values,
                "^:",
                color="#ff7f0e",
                lw=1.6,
                ms=5,
                label=f"{cli_neg} (silhueta negativa k=3: {sil_neg:.4f})",
            )
    axes[1, 0].axhline(0.0, color="#555555", lw=0.8, linestyle="--")
    axes[1, 0].set_title("Perfil Sazonal Mensal Médio (Z-score) — Ward k=3 e Fronteira")
    axes[1, 0].set_xlabel("Mês do ano")
    axes[1, 0].set_ylabel("Z-score médio mensal")
    axes[1, 0].set_xticks(meses)
    axes[1, 0].set_xticklabels(nomes_meses)
    axes[1, 0].set_ylim(-0.78, 1.48)
    axes[1, 0].legend(loc="upper left", fontsize=8.3)

    # Painel D: Perfil horário médio (0h–23h) dos medóides e grupos
    horas = np.arange(0, 24)
    df_h_norm = normalizar_zscore_por_serie(df_horario)
    perfil_h_grupos = df_h_norm.groupby(df_h_norm.index.hour).mean()
    for c_id in sorted(np.unique(rot_w3).tolist()):
        cols_c = df_diario.columns[rot_w3 == c_id]
        curva_h = perfil_h_grupos[cols_c].mean(axis=1)
        axes[1, 1].plot(
            horas,
            curva_h.values,
            "o-",
            color=cores_w3.get(c_id, "#9467bd"),
            lw=2.0,
            ms=4,
            label=f"C{c_id} média do grupo (n={len(cols_c)})",
        )
    medoide_majoritario = max(
        resultados_agnes["average"]["avaliacoes_k"][2]["clusters"],
        key=lambda c: c["tamanho"],
    )["medoide"]
    axes[1, 1].plot(
        horas,
        perfil_h_grupos[medoide_majoritario].values,
        "s--",
        color="#000000",
        lw=1.6,
        ms=4,
        label=f"{medoide_majoritario} (medóide do cluster majoritário)",
    )
    axes[1, 1].axhline(0.0, color="#555555", lw=0.8, linestyle="--")
    axes[1, 1].set_title(f"Perfil Intradiário Médio (0h–23h) dos Grupos e Medóide {medoide_majoritario}")
    axes[1, 1].set_xlabel("Hora do dia (h)")
    axes[1, 1].set_ylabel("Consumo horário padronizado (Z-score)")
    axes[1, 1].set_xticks(np.arange(0, 24, 2))
    axes[1, 1].set_ylim(-1.70, 1.55)
    axes[1, 1].legend(loc="upper left", fontsize=8.3)

    plt.tight_layout()
    p_perfis = dir_fig / "etapa1_perfis_clusters.png"
    fig.savefig(p_perfis, bbox_inches="tight")
    plt.close(fig)
    arquivos_salvos.append(str(p_perfis))

    return arquivos_salvos


def salvar_saidas_agrupamento(
    estatisticas_dtw: dict[str, Any],
    resultados_agnes: dict[str, Any],
    df_perfis: pd.DataFrame,
    df_clientes_rotulos: pd.DataFrame,
    figuras: list[str],
    dir_resultados: Path | str = DIRETORIO_RESULTADOS_PADRAO,
) -> dict[str, str]:
    """Persiste métricas estruturadas em JSON e tabelas em CSV em Resultados/."""
    p_res = Path(dir_resultados)
    p_res.mkdir(parents=True, exist_ok=True)

    metricas_serializaveis: dict[str, Any] = {
        "semente": SEMENTE_PADRAO,
        "dtw": estatisticas_dtw,
        "metodos": {},
        "figuras_geradas": figuras,
    }
    for metodo, dados in resultados_agnes.items():
        metricas_serializaveis["metodos"][metodo] = {
            str(k): v for k, v in dados["avaliacoes_k"].items()
        }

    arq_json = p_res / "metricas_agrupamento.json"
    arq_perfis = p_res / "resumo_clusters_etapa1.csv"
    arq_rotulos = p_res / "rotulos_clientes_agrupamento.csv"

    with open(arq_json, "w", encoding="utf-8") as f:
        json.dump(metricas_serializaveis, f, ensure_ascii=False, indent=2)

    df_perfis.to_csv(arq_perfis, index=False)
    df_clientes_rotulos.to_csv(arq_rotulos, index=False)

    return {
        "metricas_json": str(arq_json),
        "resumo_clusters_csv": str(arq_perfis),
        "rotulos_clientes_csv": str(arq_rotulos),
    }
