"""Script principal da Etapa 1 — Agrupamento: AGNES + DTW + Coeficiente de Silhueta.

Executa de ponta a ponta:
1. Carrega os painéis tratados (diário '1D' e horário '1h') dos 154 clientes completos.
2. Normaliza as séries diárias em Z-score por cliente e calcula (ou carrega em cache)
   a matriz de distâncias L2-DTW com banda de Sakoe-Chiba (r = 7 dias).
3. Executa AGNES com ligações 'average' e 'ward' para k = 2..10, calculando
   Coeficiente de Silhueta global, por cluster e individual, além dos medóides.
4. Consolida perfis físicos dos clusters e gera as 4 figuras em Documentacao/figuras/.
5. Salva métricas em JSON/CSV em Resultados/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from Codigo.src.agrupamento import (  # noqa: E402
    INTERVALO_K,
    METODOS_LIGACAO,
    RAIO_SAKOE_CHIBA_DIAS,
    SEMENTE_PADRAO,
    avaliar_agnes_silhueta,
    calcular_ou_carregar_matriz_dtw,
    gerar_figuras_agrupamento,
    gerar_tabela_clientes_agrupamento,
    normalizar_zscore_por_serie,
    resumir_perfis_clusters,
    salvar_saidas_agrupamento,
)
from Codigo.src.preparacao import carregar_serie_tratada  # noqa: E402


def executar_pipeline_agrupamento(forcar_recalculo_dtw: bool = False) -> dict:
    """Executa todas as subetapas da Etapa 1 e retorna o resumo de métricas."""
    np.random.seed(SEMENTE_PADRAO)

    print("[1/5] Carregando séries tratadas (diária 'D' e horária 'h')...")
    df_diario = carregar_serie_tratada(
        RAIZ_PROJETO / "Dados/tratado/consumo_diario_clientes_completos_2011_2014.parquet",
        freq="D",
    )
    df_horario = carregar_serie_tratada(
        RAIZ_PROJETO / "Dados/tratado/consumo_horario_clientes_completos_2011_2014.parquet",
        freq="h",
    )
    print(
        f"      Diário: {df_diario.shape} (freq={df_diario.index.freqstr}) | "
        f"Horário: {df_horario.shape} (freq={df_horario.index.freqstr})"
    )

    print(
        f"[2/5] Padronizando Z-score e calculando/carregando matriz DTW (Sakoe-Chiba r={RAIO_SAKOE_CHIBA_DIAS})..."
    )
    df_z = normalizar_zscore_por_serie(df_diario)
    caminho_cache = RAIZ_PROJETO / "Resultados/matriz_dtw_diario_154.npy"
    matriz_dtw, est_dtw = calcular_ou_carregar_matriz_dtw(
        df_z,
        raio=RAIO_SAKOE_CHIBA_DIAS,
        caminho_cache=caminho_cache,
        forcar_recalculo=forcar_recalculo_dtw,
    )
    est_dtw["caminho_cache"] = str(Path(est_dtw["caminho_cache"]).relative_to(RAIZ_PROJETO))
    print(
        f"      Cache usado: {est_dtw['carregado_do_cache']} | Tempo cálculo: {est_dtw['tempo_calculo_seg']}s | "
        f"Mediana DTW: {est_dtw['distancia_mediana']}"
    )

    print("[3/5] Executando AGNES (average e ward) e avaliando Silhueta para k=2..10...")
    nomes_clientes = list(df_diario.columns)
    resultados_agnes = avaliar_agnes_silhueta(
        matriz_dtw=matriz_dtw,
        nomes_clientes=nomes_clientes,
        metodos=METODOS_LIGACAO,
        ks=INTERVALO_K,
    )
    for metodo in METODOS_LIGACAO:
        print(f"      --- Ligação: {metodo} ---")
        for k in INTERVALO_K:
            av = resultados_agnes[metodo]["avaliacoes_k"][k]
            print(
                f"        k={k:2d} | silhueta={av['silhueta_media']:.4f} | "
                f"negativos={av['n_negativos']:2d} | tamanhos={av['tamanhos_clusters']}"
            )

    print("[4/5] Resumindo perfis físicos dos clusters e rótulos por cliente...")
    df_perfis = pd.concat(
        [
            resumir_perfis_clusters(
                df_diario,
                df_horario,
                matriz_dtw,
                resultados_agnes["average"]["rotulos_por_k"][2],
                resultados_agnes["average"]["silhuetas_por_k"][2],
                nome_particao="average_k2",
            ),
            resumir_perfis_clusters(
                df_diario,
                df_horario,
                matriz_dtw,
                resultados_agnes["ward"]["rotulos_por_k"][3],
                resultados_agnes["ward"]["silhuetas_por_k"][3],
                nome_particao="ward_k3",
            ),
            resumir_perfis_clusters(
                df_diario,
                df_horario,
                matriz_dtw,
                resultados_agnes["ward"]["rotulos_por_k"][4],
                resultados_agnes["ward"]["silhuetas_por_k"][4],
                nome_particao="ward_k4",
            ),
        ],
        ignore_index=True,
    )
    df_clientes_rotulos = gerar_tabela_clientes_agrupamento(df_diario, resultados_agnes)

    print("[5/5] Gerando figuras e persistindo saídas em Resultados/...")
    figuras = gerar_figuras_agrupamento(
        df_diario=df_diario,
        df_horario=df_horario,
        resultados_agnes=resultados_agnes,
        diretorio_figuras=RAIZ_PROJETO / "Documentacao/figuras",
    )
    figuras_rel = [str(Path(f).relative_to(RAIZ_PROJETO)) for f in figuras]

    saidas = salvar_saidas_agrupamento(
        estatisticas_dtw=est_dtw,
        resultados_agnes=resultados_agnes,
        df_perfis=df_perfis,
        df_clientes_rotulos=df_clientes_rotulos,
        figuras=figuras_rel,
        dir_resultados=RAIZ_PROJETO / "Resultados",
    )
    print("      Arquivos persistidos:")
    for chave, valor in saidas.items():
        print(f"        - {chave}: {Path(valor).relative_to(RAIZ_PROJETO)}")

    return {
        "dtw": est_dtw,
        "figuras": figuras_rel,
        "saidas": saidas,
    }


if __name__ == "__main__":
    executar_pipeline_agrupamento()
    print("\nEtapa 1 concluída com sucesso.")
