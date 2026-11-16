"""Script principal da Etapa 0 — Preparação e tratamento dos dados.

Executa de ponta a ponta:
1. Carregamento e auditoria dos dados brutos (sem alterar Dados/bruto/).
2. Tratamento de zeros pré-adesão/pós-desligamento, DST (março/outubro) e falhas curtas.
3. Regularização temporal para grades '15min', '1h' e '1D' e seleção de clientes completos.
4. Geração de figuras antes/depois em Documentacao/figuras/.
5. Persistência dos dados tratados em Dados/tratado/ e métricas em Resultados/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Garante execução reprodutível independente do diretório de invocação
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROJETO))

from Codigo.src.preparacao import (  # noqa: E402
    SEMENTE_PADRAO,
    agregar_e_filtrar_series,
    carregar_dados_brutos,
    diagnosticar_dados_brutos,
    gerar_figuras_preparacao,
    salvar_saidas_preparacao,
    tratar_series_15min,
)


def executar_pipeline_preparacao() -> dict:
    """Executa todas as subetapas da preparação e retorna as métricas consolidadas."""
    np.random.seed(SEMENTE_PADRAO)

    caminho_bruto = RAIZ_PROJETO / "Dados/bruto/LD2011_2014.txt"
    print(f"[1/5] Carregando base bruta: {caminho_bruto.relative_to(RAIZ_PROJETO)}")
    df_bruto = carregar_dados_brutos(caminho_bruto)
    print(
        f"      Dimensões: {df_bruto.shape} | Período: {df_bruto.index.min()} a {df_bruto.index.max()} | freq={df_bruto.index.freqstr}"
    )

    print("[2/5] Auditando dados brutos...")
    diag_bruto = diagnosticar_dados_brutos(df_bruto)
    print(
        f"      Nulos explícitos: {diag_bruto['nulos_explicitos']} | Duplicatas: {diag_bruto['duplicatas_timestamp']} | Negativos: {diag_bruto['valores_negativos']}"
    )
    print(
        f"      Zeros totais: {diag_bruto['zeros_totais']:,} ({diag_bruto['pct_zeros_totais']:.2f}%) | "
        f"Pré-adesão: {diag_bruto['zeros_pre_adesao']:,} | Pós-desligamento: {diag_bruto['zeros_pos_desligamento']:,} | "
        f"Internos: {diag_bruto['zeros_internos']:,}"
    )

    print("[3/5] Tratando zeros estruturais, DST (março e outubro) e interpolando falhas curtas...")
    df_tratado_15min, df_resumo_clientes, resumo_tratamento = tratar_series_15min(df_bruto)
    print(
        f"      DST março -> NaN: {resumo_tratamento['celulas_dst_marco_identificadas']:,} | "
        f"DST outubro (/2): {resumo_tratamento['celulas_dst_outubro_divididas_por_2']:,} | "
        f"Zeros internos curtos: {resumo_tratamento['celulas_zeros_internos_curtos_nan']:,}"
    )
    print(
        f"      Células interpoladas (grade 15min): {resumo_tratamento['celulas_interpoladas_total']:,} "
        f"({resumo_tratamento['pct_celulas_interpoladas_sobre_base_total']:.6f}% da base) | "
        f"NaNs na janela ativa após tratamento: {resumo_tratamento['nans_remanescentes_janela_ativa']}"
    )

    print("[4/5] Regularizando para frequências horária ('1h') e diária ('1D') e selecionando clientes completos...")
    agregado = agregar_e_filtrar_series(df_tratado_15min, df_resumo_clientes)
    res_ag = agregado["resumo_agregacao"]
    print(
        f"      Clientes na janela 2011-2014: {res_ag['n_clientes_janela_2011_2014']} | "
        f"Selecionados completos (<= 1% zeros longos): {res_ag['n_clientes_completos_selecionados']} | "
        f"Excluídos por zeros longos: {res_ag['clientes_excluidos_por_zeros_longos']}"
    )
    print(
        f"      Shapes -> Horário completos: {res_ag['shape_horario_completos']} (freq={res_ag['freq_horario']}) | "
        f"Diário completos: {res_ag['shape_diario_completos']} (freq={res_ag['freq_diario']})"
    )

    print("[5/5] Gerando figuras e salvando arquivos tratados/métricas...")
    figuras = gerar_figuras_preparacao(
        df_bruto=df_bruto,
        df_tratado_15min=df_tratado_15min,
        df_horario_completos=agregado["df_horario_completos"],
        df_diario_completos=agregado["df_diario_completos"],
        df_resumo_clientes=df_resumo_clientes,
        diretorio_figuras=RAIZ_PROJETO / "Documentacao/figuras",
    )

    metricas = {
        "semente": SEMENTE_PADRAO,
        "diagnostico_bruto": diag_bruto,
        "tratamento_15min": resumo_tratamento,
        "agregacao_e_selecao": res_ag,
        "figuras_geradas": [str(Path(f).relative_to(RAIZ_PROJETO)) for f in figuras],
    }

    saidas = salvar_saidas_preparacao(
        agregado=agregado,
        df_resumo_clientes=df_resumo_clientes,
        metricas=metricas,
        dir_tratado=RAIZ_PROJETO / "Dados/tratado",
        dir_resultados=RAIZ_PROJETO / "Resultados",
    )
    print("      Arquivos persistidos:")
    for chave, valor in saidas.items():
        print(f"        - {chave}: {Path(valor).relative_to(RAIZ_PROJETO)}")

    return metricas


if __name__ == "__main__":
    resultado = executar_pipeline_preparacao()
    print("\nResumo JSON concluído com sucesso.")
