"""Script principal da Etapa 2 — Descoberta de Padrões e Anomalias com Matrix Profile.

Executa de ponta a ponta:
1. Carrega as séries tratadas (horária 'h' e diária 'D') dos 154 clientes completos e extrai
   a série representativa MT_206 (medóide do cluster majoritário da Etapa 1).
2. Calcula (ou carrega em cache) o Matrix Profile via `stumpy.stump` com normalização Z-score
   em três janelas fisicamente justificadas: m=168h (semanal, principal), m=24h (diária)
   e m=7 dias (semanal na escala diária).
3. Extrai os top-3 motifs e top-3 discords não sobrepostos para cada janela.
4. Executa o estudo comparativo de normalização Z-score (`normalize=True` vs. `normalize=False`)
   na série real e em um experimento controlado com transformação afim (escala + offset).
5. Gera as 4 figuras diagnósticas em Documentacao/figuras/ e salva JSON/CSV em Resultados/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from Codigo.src.matrix_profile import (  # noqa: E402
    CLIENTE_REPRESENTATIVO,
    JANELA_DIARIA_SEMANAL_M,
    JANELA_HORARIA_DIARIA_M,
    JANELA_HORARIA_SEMANAL_M,
    SEMENTE_PADRAO,
    TOP_K_PADROES,
    calcular_ou_carregar_mp,
    executar_experimento_zscore,
    extrair_top_discords,
    extrair_top_motifs,
    gerar_figuras_matrix_profile,
    salvar_saidas_matrix_profile,
)
from Codigo.src.preparacao import carregar_serie_tratada  # noqa: E402


def executar_pipeline_matrix_profile(forcar_recalculo: bool = False) -> dict:
    """Executa todas as subetapas da Etapa 2 e retorna o resumo de métricas."""
    np.random.seed(SEMENTE_PADRAO)

    print(f"[1/5] Carregando série representativa {CLIENTE_REPRESENTATIVO} (horária 'h' e diária 'D')...")
    df_horario = carregar_serie_tratada(
        RAIZ_PROJETO / "Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet",
        freq="h",
    )
    df_diario = carregar_serie_tratada(
        RAIZ_PROJETO / "Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet",
        freq="D",
    )
    serie_h = df_horario[CLIENTE_REPRESENTATIVO].copy()
    serie_d = df_diario[CLIENTE_REPRESENTATIVO].copy()

    resumo_serie = {
        "cliente": CLIENTE_REPRESENTATIVO,
        "n_horas": int(len(serie_h)),
        "n_dias": int(len(serie_d)),
        "inicio": str(serie_h.index.min()),
        "fim": str(serie_h.index.max()),
        "nans_horario": int(serie_h.isna().sum()),
        "nans_diario": int(serie_d.isna().sum()),
        "media_kw": round(float(serie_h.mean()), 2),
        "desvio_kw": round(float(serie_h.std()), 2),
        "min_kw": round(float(serie_h.min()), 2),
        "mediana_kw": round(float(serie_h.median()), 2),
        "max_kw": round(float(serie_h.max()), 2),
    }
    print(
        f"      Horas: {resumo_serie['n_horas']} | Dias: {resumo_serie['n_dias']} | "
        f"Média: {resumo_serie['media_kw']} kW | Desvio: {resumo_serie['desvio_kw']} kW"
    )

    print("[2/5] Calculando/carregando Matrix Profile multi-escala (m=168h, m=24h, m=7d)...")
    mp168_dist, mp168_idx, est_168 = calcular_ou_carregar_mp(
        serie_h,
        m=JANELA_HORARIA_SEMANAL_M,
        normalizar=True,
        caminho_cache=RAIZ_PROJETO / "Resultados/mp_horario_m168_zscore.npy",
        forcar_recalculo=forcar_recalculo,
    )
    mp24_dist, mp24_idx, est_24 = calcular_ou_carregar_mp(
        serie_h,
        m=JANELA_HORARIA_DIARIA_M,
        normalizar=True,
        caminho_cache=RAIZ_PROJETO / "Resultados/mp_horario_m24_zscore.npy",
        forcar_recalculo=forcar_recalculo,
    )
    mp7d_dist, mp7d_idx, est_7d = calcular_ou_carregar_mp(
        serie_d,
        m=JANELA_DIARIA_SEMANAL_M,
        normalizar=True,
        caminho_cache=RAIZ_PROJETO / "Resultados/mp_diario_m7_zscore.npy",
        forcar_recalculo=forcar_recalculo,
    )
    for est in (est_168, est_24, est_7d):
        if est["caminho_cache"] is not None:
            est["caminho_cache"] = str(Path(est["caminho_cache"]).relative_to(RAIZ_PROJETO))

    estatisticas_mp = {
        "horario_m168": est_168,
        "horario_m24": est_24,
        "diario_m7": est_7d,
    }
    for chave, est in estatisticas_mp.items():
        print(
            f"      {chave:13s} | cache={str(est['carregado_do_cache']):5s} | "
            f"t={est['tempo_calculo_seg']:5.2f}s | min={est['dist_min']:.4f} | "
            f"med={est['dist_mediana']:.4f} | max={est['dist_max']:.4f}"
        )

    print(f"[3/5] Extraindo top-{TOP_K_PADROES} motifs e discords não sobrepostos...")
    motifs_por_janela = {
        "horario_m168": extrair_top_motifs(serie_h, mp168_dist, mp168_idx, JANELA_HORARIA_SEMANAL_M, TOP_K_PADROES),
        "horario_m24": extrair_top_motifs(serie_h, mp24_dist, mp24_idx, JANELA_HORARIA_DIARIA_M, TOP_K_PADROES),
        "diario_m7": extrair_top_motifs(serie_d, mp7d_dist, mp7d_idx, JANELA_DIARIA_SEMANAL_M, TOP_K_PADROES),
    }
    discords_por_janela = {
        "horario_m168": extrair_top_discords(serie_h, mp168_dist, mp168_idx, JANELA_HORARIA_SEMANAL_M, TOP_K_PADROES),
        "horario_m24": extrair_top_discords(serie_h, mp24_dist, mp24_idx, JANELA_HORARIA_DIARIA_M, TOP_K_PADROES),
        "diario_m7": extrair_top_discords(serie_d, mp7d_dist, mp7d_idx, JANELA_DIARIA_SEMANAL_M, TOP_K_PADROES),
    }
    for esc in ("horario_m168", "horario_m24", "diario_m7"):
        m1 = motifs_por_janela[esc][0]
        d1 = discords_por_janela[esc][0]
        print(
            f"      {esc:13s} | Motif #1: {m1['inicio_a'][:13]} <-> {m1['inicio_b'][:13]} "
            f"(d={m1['distancia_mp']:.4f}, r={m1['correlacao_pearson']:.4f})"
        )
        print(
            f"      {'':13s} | Discord #1: {d1['inicio_discord'][:13]} "
            f"(d={d1['distancia_mp']:.4f}, viz={d1['inicio_vizinho'][:13]}, r={d1['correlacao_com_vizinho']:.4f})"
        )

    print("[4/5] Executando análise comparativa do papel da normalização Z-score...")
    exp_zscore = executar_experimento_zscore(
        serie_horaria=serie_h,
        m=JANELA_HORARIA_SEMANAL_M,
        idx_alvo=motifs_por_janela["horario_m168"][0]["idx_a"],
        mp_norm_precomp=(mp168_dist, mp168_idx),
        caminho_cache_sem_norm=RAIZ_PROJETO / "Resultados/mp_horario_m168_sem_norm.npy",
    )
    orig_z = exp_zscore["serie_original"]
    pert_z = exp_zscore["experimento_perturbacao"]
    print(
        f"      Corr(MP sem norm, σ_local) = {orig_z['corr_mp_sem_norm_com_desvio_local']:.4f} | "
        f"Corr(MP Z-score, σ_local) = {orig_z['corr_mp_zscore_com_desvio_local']:.4f}"
    )
    print(
        f"      Perturbação afim: sem norm saltou {pert_z['razao_salto_sem_norm']:.2f}x "
        f"(rank discord #{pert_z['rank_discord_pert_sem_norm']}) | "
        f"com Z-score Δ={pert_z['delta_abs_zscore']:.6f} (mantém Motif #{pert_z['rank_motif_pert_zscore']})"
    )

    print("[5/5] Gerando figuras e persistindo saídas em Resultados/...")
    figuras = gerar_figuras_matrix_profile(
        serie_horaria=serie_h,
        serie_diaria=serie_d,
        mp_168=(mp168_dist, mp168_idx),
        mp_24=(mp24_dist, mp24_idx),
        mp_7d=(mp7d_dist, mp7d_idx),
        motifs_168=motifs_por_janela["horario_m168"],
        motifs_24=motifs_por_janela["horario_m24"],
        motifs_7d=motifs_por_janela["diario_m7"],
        discords_168=discords_por_janela["horario_m168"],
        discords_24=discords_por_janela["horario_m24"],
        exp_zscore=exp_zscore,
        diretorio_figuras=RAIZ_PROJETO / "Documentacao/figuras",
    )
    figuras_rel = [str(Path(f).relative_to(RAIZ_PROJETO)) for f in figuras]

    saidas = salvar_saidas_matrix_profile(
        cliente=CLIENTE_REPRESENTATIVO,
        resumo_serie=resumo_serie,
        estatisticas_mp=estatisticas_mp,
        motifs_por_janela=motifs_por_janela,
        discords_por_janela=discords_por_janela,
        exp_zscore=exp_zscore,
        figuras=figuras_rel,
        dir_resultados=RAIZ_PROJETO / "Resultados",
    )
    print("      Arquivos persistidos:")
    for chave, valor in saidas.items():
        print(f"        - {chave}: {Path(valor).relative_to(RAIZ_PROJETO)}")

    return {
        "estatisticas_mp": estatisticas_mp,
        "figuras": figuras_rel,
        "saidas": saidas,
    }


if __name__ == "__main__":
    executar_pipeline_matrix_profile()
    print("\nEtapa 2 concluída com sucesso.")
