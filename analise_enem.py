"""
Análise Exploratória e Estatística do ENEM 2023
Relação entre renda familiar e desempenho dos candidatos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Configurações de exibição
sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = Path('outputs')
DATA_PATH = Path('MICRODADOS_ENEM_2023.csv')
RENDAS = list('ABCDEFGHIJKLMNOPQ')
RANK_RENDA = {letra: idx + 1 for idx, letra in enumerate(RENDAS)}

RENDA_FAIXAS_REAIS = {
    'A': 'Nenhuma Renda',
    'B': 'Até R$ 1.320,00',
    'C': 'De R$ 1.320,01 até R$ 1.980,00',
    'D': 'De R$ 1.980,01 até R$ 2.640,00',
    'E': 'De R$ 2.640,01 até R$ 3.300,00',
    'F': 'De R$ 3.300,01 até R$ 3.960,00',
    'G': 'De R$ 3.960,01 até R$ 5.280,00',
    'H': 'De R$ 5.280,01 até R$ 6.600,00',
    'I': 'De R$ 6.600,01 até R$ 7.920,00',
    'J': 'De R$ 7.920,01 até R$ 9.240,00',
    'K': 'De R$ 9.240,01 até R$ 10.560,00',
    'L': 'De R$ 10.560,01 até R$ 11.880,00',
    'M': 'De R$ 11.880,01 até R$ 13.200,00',
    'N': 'De R$ 13.200,01 até R$ 15.840,00',
    'O': 'De R$ 15.840,01 até R$ 19.800,00',
    'P': 'De R$ 19.800,01 até R$ 26.400,00',
    'Q': 'Mais de R$ 26.400,00',
}

RENDA_NUMERICA = {
    'A': 0.0,
    'B': 660.0,
    'C': 1650.0,
    'D': 2310.0,
    'E': 2970.0,
    'F': 3630.0,
    'G': 4620.0,
    'H': 5940.0,
    'I': 7260.0,
    'J': 8580.0,
    'K': 9900.0,
    'L': 11220.0,
    'M': 12540.0,
    'N': 14520.0,
    'O': 17820.0,
    'P': 23100.0,
    'Q': 30000.0,
}

ORDEM_GRUPO_RENDA = ['Baixa', 'Média-baixa', 'Média', 'Média-alta', 'Alta']
COLUNAS_NOTAS = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
PROVAS_NOMES = {
    'NU_NOTA_CN': 'Ciências Natureza',
    'NU_NOTA_CH': 'Ciências Humanas',
    'NU_NOTA_LC': 'Linguagens',
    'NU_NOTA_MT': 'Matemática',
    'NU_NOTA_REDACAO': 'Redação',
}
ESCOLA_MAP = {1: 'Não Respondeu', 2: 'Pública', 3: 'Privada'}
RACA_MAP = {0: 'Não Declarada', 1: 'Branca', 2: 'Preta', 3: 'Parda', 4: 'Amarela', 5: 'Indígena'}
UF_PARA_REGIAO = {
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste', 'PB': 'Nordeste', 'PE': 'Nordeste',
    'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    'DF': 'Centro-Oeste', 'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul',
}


def criar_pasta_saida(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def gerar_dados_sinteticos(caminho_arquivo: Path, n_linhas: int = 150000, random_state: int = 42) -> None:
    print(f'[Info] Gerando {n_linhas:,} registros sintéticos do ENEM...')
    np.random.seed(random_state)

    sexo = np.random.choice(['F', 'M'], size=n_linhas, p=[0.60, 0.40])
    ufs = list(UF_PARA_REGIAO.keys())
    p_ufs = np.array([
        0.18, 0.10, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03,
        0.02, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
    ])
    p_ufs = p_ufs / p_ufs.sum()
    sg_ufs = np.random.choice(ufs, size=n_linhas, p=p_ufs)

    racas = [0, 1, 2, 3, 4, 5]
    p_racas = [0.02, 0.35, 0.12, 0.48, 0.02, 0.01]
    tp_cor_raca = np.random.choice(racas, size=n_linhas, p=p_racas)
    in_treineiro = np.random.choice([0, 1], size=n_linhas, p=[0.85, 0.15])

    pse_score = np.random.normal(loc=0.0, scale=1.0, size=n_linhas)
    pse_score += np.where(tp_cor_raca == 1, 0.5, 0.0)
    pse_score += np.where(tp_cor_raca == 4, 0.6, 0.0)
    pse_score -= np.where(tp_cor_raca == 2, 0.3, 0.0)
    pse_score -= np.where(tp_cor_raca == 3, 0.2, 0.0)
    pse_score -= np.where(tp_cor_raca == 5, 0.6, 0.0)

    q006_array = np.empty(n_linhas, dtype=object)
    sorted_idx = np.argsort(pse_score)
    limites = np.round(np.cumsum(np.array([
        0.05, 0.25, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
        0.02, 0.015, 0.015, 0.01, 0.005, 0.003, 0.002
    ]) * n_linhas)).astype(int)
    limites[-1] = n_linhas

    start = 0
    for letra, end in zip(RENDAS, limites):
        q006_array[sorted_idx[start:end]] = letra
        start = end

    tp_escola = np.ones(n_linhas, dtype=int)
    mask_privada = np.isin(q006_array, ['N', 'O', 'P', 'Q'])
    mask_mista = np.isin(q006_array, ['J', 'K', 'L', 'M'])
    mask_intermediaria = np.isin(q006_array, ['G', 'H', 'I'])
    rest = ~(mask_privada | mask_mista | mask_intermediaria)
    tp_escola[mask_privada] = np.random.choice([2, 3], size=mask_privada.sum(), p=[0.15, 0.85])
    tp_escola[mask_mista] = np.random.choice([2, 3], size=mask_mista.sum(), p=[0.50, 0.50])
    tp_escola[mask_intermediaria] = np.random.choice([2, 3], size=mask_intermediaria.sum(), p=[0.80, 0.20])
    tp_escola[rest] = np.random.choice([1, 2, 3], size=rest.sum(), p=[0.05, 0.92, 0.03])

    renda_efeito = np.array([RENDA_NUMERICA[letra] for letra in q006_array]) * 15.0
    escola_efeito = np.where(tp_escola == 3, 60.0, 0.0) + np.where(tp_escola == 2, -15.0, 0.0)
    nota_base = 450.0 + renda_efeito + escola_efeito

    nu_nota_cn = np.clip(np.random.normal(nota_base * 0.90, 65.0), 0, 1000)
    nu_nota_ch = np.clip(np.random.normal(nota_base * 1.05, 60.0), 0, 1000)
    nu_nota_lc = np.clip(np.random.normal(nota_base * 1.02, 55.0), 0, 1000)
    nu_nota_mt = np.clip(np.random.normal(nota_base * 0.95 + 15.0, 80.0), 0, 1000)
    nu_nota_redacao = np.clip(np.random.normal(nota_base * 1.10 + 30.0, 130.0), 0, 1000)

    faltosos = np.random.choice([True, False], size=n_linhas, p=[0.08, 0.92])
    nu_nota_cn[faltosos] = np.nan
    nu_nota_mt[faltosos] = np.nan
    faltosos_dia2 = np.random.choice([True, False], size=n_linhas, p=[0.07, 0.93])
    nu_nota_ch[faltosos_dia2] = np.nan
    nu_nota_lc[faltosos_dia2] = np.nan
    nu_nota_redacao[faltosos_dia2] = np.nan

    df_sint = pd.DataFrame({
        'IN_TREINEIRO': in_treineiro,
        'Q006': q006_array,
        'NU_NOTA_CN': nu_nota_cn,
        'NU_NOTA_CH': nu_nota_ch,
        'NU_NOTA_LC': nu_nota_lc,
        'NU_NOTA_MT': nu_nota_mt,
        'NU_NOTA_REDACAO': nu_nota_redacao,
        'SG_UF_PROVA': sg_ufs,
        'TP_COR_RACA': tp_cor_raca,
        'TP_SEXO': sexo,
        'TP_ESCOLA': tp_escola,
    })
    df_sint.to_csv(caminho_arquivo, sep=';', index=False, encoding='latin-1')
    print(f'[Sucesso] Base sintética salva em {caminho_arquivo}.')


def carregar_dados(caminho_arquivo: Path) -> pd.DataFrame:
    if not caminho_arquivo.exists():
        gerar_dados_sinteticos(caminho_arquivo)
    print(f'[Info] Carregando dados de {caminho_arquivo}...')
    df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1', low_memory=False)
    print(f'[Sucesso] Dados carregados: {df.shape[0]:,} registros e {df.shape[1]} colunas.')
    return df


def categorizar_grupo_renda(valor: str) -> str:
    if valor in ['A', 'B', 'C']:
        return 'Baixa'
    if valor in ['D', 'E', 'F']:
        return 'Média-baixa'
    if valor in ['G', 'H', 'I']:
        return 'Média'
    if valor in ['J', 'K', 'L', 'M']:
        return 'Média-alta'
    if valor in ['N', 'O', 'P', 'Q']:
        return 'Alta'
    return 'Indefinido'


def filtrar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    total_original = len(df)
    df = df[df['IN_TREINEIRO'] == 0]
    total_filtrado = len(df)
    print(f'[Filtro] Removidos {total_original - total_filtrado:,} treineiros.')
    nulos = df[COLUNAS_NOTAS].isna().sum()
    print('[Info] Nulos antes do filtro:')
    for coluna, quantidade in nulos.items():
        print(f'  - {coluna}: {quantidade:,} ({quantidade / total_filtrado * 100:.2f}%)')
    df = df.dropna(subset=COLUNAS_NOTAS).copy()
    total_completo = len(df)
    print(f'[Filtro] Removidos {total_filtrado - total_completo:,} registros com notas ausentes.')
    print(f'[Info] Total após limpeza: {total_completo:,} registros ({total_completo / total_original * 100:.2f}% do original).')
    return df


def amostrar_dados(df: pd.DataFrame, frac: float = 0.20) -> pd.DataFrame:
    if len(df) > 200_000:
        print(f'[Amostragem] Amostrando 20% em Q006 ({len(df):,} registros).')
        amostra = df.groupby('Q006', group_keys=False).apply(
            lambda subset: subset.sample(frac=frac, random_state=42)
        )
        print(f'[Amostragem] Base amostrada: {len(amostra):,} registros.')
        return amostra.copy()
    print(f'[Amostragem] Não necessária ({len(df):,} registros).')
    return df.copy()


def criar_variaveis_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['RENDA_DESCRICAO'] = df['Q006'].map(RENDA_FAIXAS_REAIS)
    df['RENDA_NUMERICA'] = df['Q006'].map(RENDA_NUMERICA)
    df['NOTA_MEDIA'] = df[COLUNAS_NOTAS].mean(axis=1)
    df['grupo_renda'] = df['Q006'].apply(categorizar_grupo_renda)
    df['ESCOLA_DESCRICAO'] = df['TP_ESCOLA'].map(ESCOLA_MAP)
    df['RACA_DESCRICAO'] = df['TP_COR_RACA'].map(RACA_MAP)
    df['REGIAO'] = df['SG_UF_PROVA'].map(UF_PARA_REGIAO)
    df['RENDA_RANK'] = df['Q006'].map(RANK_RENDA)
    print('[Info] Variáveis derivadas criadas.')
    return df


def imprimir_estatisticas_iniciais(df: pd.DataFrame) -> None:
    print('\n--- Estatísticas Iniciais ---')
    print(f'Shape: {df.shape}')
    print('Nulos nas notas:')
    print(df[COLUNAS_NOTAS].isna().sum())
    dist = df['Q006'].value_counts().sort_index()
    dist_pct = df['Q006'].value_counts(normalize=True).sort_index() * 100
    for letra, quantidade in dist.items():
        print(f'  Faixa {letra} ({RENDA_FAIXAS_REAIS[letra]}): {quantidade:,} ({dist_pct[letra]:.2f}%)')


def analise_por_faixa_renda(df: pd.DataFrame) -> pd.DataFrame:
    print('\n--- Estatísticas por Faixa de Renda ---')
    estatisticas = df.groupby('Q006')['NOTA_MEDIA'].agg(
        Media='mean',
        Mediana='median',
        Desvio_Padrao='std',
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
        Qtd='count',
    ).sort_index()
    for idx, row in estatisticas.iterrows():
        print(
            f'Faixa {idx} ({RENDA_FAIXAS_REAIS[idx]}): '
            f'Média={row.Media:.1f} | Mediana={row.Mediana:.1f} | '
            f'DP={row.Desvio_Padrao:.1f} | IQR=[{row.P25:.1f} - {row.P75:.1f}] | n={int(row.Qtd):,}'
        )
    return estatisticas


def tabela_cruzada_renda_escola(df: pd.DataFrame) -> None:
    print('\n--- Tabela cruzada: Grupo de renda vs Escola ---')
    cruzada = pd.crosstab(df['grupo_renda'], df['ESCOLA_DESCRICAO'])
    cruzada_pct = pd.crosstab(df['grupo_renda'], df['ESCOLA_DESCRICAO'], normalize='index') * 100
    print(cruzada)
    print('\n', cruzada_pct.round(2).astype(str) + '%')


def analise_uf_regiao(df: pd.DataFrame) -> pd.Series:
    print('\n--- Nota média por UF e Região ---')
    nota_uf = df.groupby('SG_UF_PROVA')['NOTA_MEDIA'].mean().sort_values(ascending=False)
    print('Top 5 UFs:')
    for uf, nota in nota_uf.head(5).items():
        print(f'  {uf}: {nota:.2f}')
    print('Bottom 5 UFs:')
    for uf, nota in nota_uf.tail(5).items():
        print(f'  {uf}: {nota:.2f}')
    nota_regiao = df.groupby('REGIAO')['NOTA_MEDIA'].mean().sort_values(ascending=False)
    print('\nNota média por região:')
    for regiao, nota in nota_regiao.items():
        print(f'  {regiao}: {nota:.2f}')
    return nota_regiao


def perfil_percentis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    print('\n--- Perfil demográfico: P90+ vs P10- ---')
    p90 = df['NOTA_MEDIA'].quantile(0.90)
    p10 = df['NOTA_MEDIA'].quantile(0.10)
    elite = df[df['NOTA_MEDIA'] >= p90].copy()
    vulneraveis = df[df['NOTA_MEDIA'] <= p10].copy()
    print(f'Elite P90+: >= {p90:.1f} ({len(elite):,} candidatos)')
    print(f'Vulneráveis P10-: <= {p10:.1f} ({len(vulneraveis):,} candidatos)')
    return elite, vulneraveis, p90, p10


def salvar_figure(nome: str) -> None:
    destino = OUTPUT_DIR / nome
    plt.tight_layout()
    plt.savefig(destino, dpi=150)
    plt.close()
    print(f'[Imagem] {destino.name} salvo.')


def gerar_visualizacoes(df: pd.DataFrame) -> None:
    print('\n--- Gerando visualizações ---')
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='grupo_renda',
        y='NOTA_MEDIA',
        order=ORDEM_GRUPO_RENDA,
        palette='coolwarm',
        hue='grupo_renda',
        dodge=False,
        legend=False,
    )
    plt.title('Distribuição da Nota Média por Grupo de Renda')
    plt.xlabel('Grupo de Renda Familiar')
    plt.ylabel('Nota Média das Provas')
    salvar_figure('1_boxplot_nota_por_grupo_renda.png')

    df_longo = df.melt(
        id_vars=['grupo_renda'],
        value_vars=COLUNAS_NOTAS,
        var_name='PROVA',
        value_name='NOTA',
    )
    df_longo['PROVA'] = df_longo['PROVA'].map(PROVAS_NOMES)
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df_longo,
        x='PROVA',
        y='NOTA',
        hue='grupo_renda',
        hue_order=ORDEM_GRUPO_RENDA,
        palette='coolwarm',
        errorbar=('ci', 95),
    )
    plt.title('Nota Média por Prova e Grupo de Renda')
    plt.xlabel('Área de Conhecimento')
    plt.ylabel('Nota Média (IC 95%)')
    plt.legend(title='Grupo de Renda', bbox_to_anchor=(1.02, 1), loc='upper left')
    salvar_figure('2_barplot_nota_por_prova_renda.png')

    df_grupo = df.groupby('Q006').agg(
        Nota_Media_Grupo=('NOTA_MEDIA', 'mean'),
        Renda_Media_Grupo=('RENDA_NUMERICA', 'first'),
        Candidatos=('NOTA_MEDIA', 'count'),
    ).reset_index()
    x = df['RENDA_NUMERICA']
    y = df['NOTA_MEDIA']
    x_const = sm.add_constant(x)
    modelo = sm.OLS(y, x_const).fit()
    r2 = modelo.rsquared
    p_val = modelo.pvalues.iloc[1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df.sample(n=min(2000, len(df)), random_state=42),
        x='RENDA_NUMERICA',
        y='NOTA_MEDIA',
        color='gray',
        alpha=0.12,
        label='Amostra',
    )
    sns.regplot(
        data=df,
        x='RENDA_NUMERICA',
        y='NOTA_MEDIA',
        scatter=False,
        color='red',
        line_kws={'linewidth': 2, 'label': 'Reta de regressão'},
    )
    sns.scatterplot(
        data=df_grupo,
        x='Renda_Media_Grupo',
        y='Nota_Media_Grupo',
        size='Candidatos',
        hue='Nota_Media_Grupo',
        palette='coolwarm',
        sizes=(50, 400),
        legend='brief',
        edgecolor='black',
        zorder=5,
    )
    plt.title('Relação entre Renda Estimada e Nota Média')
    plt.xlabel('Renda Familiar Estimada (R$)')
    plt.ylabel('Nota Média Geral')
    plt.text(
        x.max() * 0.55,
        y.min() + 80,
        f'R² = {r2:.4f}\np-valor = {p_val:.3e}\nCoef = {modelo.params.iloc[1] * 1000:.2f} pts / R$ 1k',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
        fontsize=11,
    )
    plt.legend(loc='lower right')
    salvar_figure('3_scatterplot_regressao_renda_nota.png')

    correlacao = df[[
        'RENDA_NUMERICA',
        'NU_NOTA_CN',
        'NU_NOTA_CH',
        'NU_NOTA_LC',
        'NU_NOTA_MT',
        'NU_NOTA_REDACAO',
        'NOTA_MEDIA',
    ]].corr(method='spearman')
    labels = ['Renda (R$)', 'C. Natureza', 'C. Humanas', 'Linguagens', 'Matemática', 'Redação', 'Média Geral']
    correlacao.columns = labels
    correlacao.index = labels
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.3f', linewidths=0.5)
    plt.title('Correlação de Spearman entre Renda e Notas')
    salvar_figure('4_heatmap_correlacao_renda_provas.png')

    df_stacked = df.groupby(['grupo_renda', 'RACA_DESCRICAO']).size().unstack(fill_value=0)
    df_stacked_pct = df_stacked.div(df_stacked.sum(axis=1), axis=0) * 100
    df_stacked_pct = df_stacked_pct.reindex(ORDEM_GRUPO_RENDA)
    plt.figure(figsize=(10, 6))
    df_stacked_pct.plot(kind='bar', stacked=True, cmap='tab10', ax=plt.gca(), edgecolor='black', width=0.6)
    plt.title('Distribuição de Cor/Raça por Grupo de Renda')
    plt.xlabel('Grupo de Renda Familiar')
    plt.ylabel('Percentual (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Cor/Raça Declarada', bbox_to_anchor=(1.02, 1), loc='upper left')
    salvar_figure('5_stacked_bar_raca_por_renda.png')

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df,
        x='grupo_renda',
        y='NU_NOTA_REDACAO',
        order=ORDEM_GRUPO_RENDA,
        palette='coolwarm',
        hue='grupo_renda',
        dodge=False,
        legend=False,
    )
    plt.title('Distribuição das Notas de Redação por Grupo de Renda')
    plt.xlabel('Grupo de Renda Familiar')
    plt.ylabel('Nota de Redação')
    salvar_figure('6_violinplot_redacao_por_grupo_renda.png')


def ajustar_modelo_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    formula = (
        'NOTA_MEDIA ~ RENDA_NUMERICA '
        '+ C(TP_ESCOLA, Treatment(reference=2)) '
        '+ C(REGIAO, Treatment(reference="Sudeste")) '
        '+ C(TP_COR_RACA, Treatment(reference=1)) '
        '+ C(TP_SEXO, Treatment(reference="F"))'
    )
    modelo = smf.ols(formula=formula, data=df).fit()
    print('\n--- Regressão OLS ---')
    print(modelo.summary())
    return modelo


def executar_testes_estatisticos(df: pd.DataFrame) -> tuple[float, float, dict[str, tuple[float, float]], dict[str, tuple[float, float]], sm.regression.linear_model.RegressionResultsWrapper]:
    print('\n--- Testes Estatísticos ---')
    grupos = [df[df['grupo_renda'] == g]['NOTA_MEDIA'].dropna().values for g in ORDEM_GRUPO_RENDA]
    f_stat, p_val_anova = stats.f_oneway(*grupos)
    print(f'ANOVA: F = {f_stat:.4f}, p = {p_val_anova:.3e}')

    correlacoes = {}
    for coluna in COLUNAS_NOTAS + ['NOTA_MEDIA']:
        nome = PROVAS_NOMES.get(coluna, 'Nota Média Geral')
        rho, p_val = stats.spearmanr(df['RENDA_RANK'], df[coluna], nan_policy='omit')
        correlacoes[nome] = (rho, p_val)
        print(f'Renda vs {nome}: rho = {rho:.4f}, p = {p_val:.3e}')

    mann_whitney = {}
    for grupo in ORDEM_GRUPO_RENDA:
        subset = df[df['grupo_renda'] == grupo]
        publica = subset[subset['TP_ESCOLA'] == 2]['NOTA_MEDIA'].dropna()
        privada = subset[subset['TP_ESCOLA'] == 3]['NOTA_MEDIA'].dropna()
        if len(publica) > 10 and len(privada) > 10:
            u_stat, p_val = stats.mannwhitneyu(publica, privada, alternative='two-sided')
            mann_whitney[grupo] = (u_stat, p_val)
            print(f'Grupo {grupo}: U={u_stat:.1f}, p={p_val:.3e}, Δ={privada.mean() - publica.mean():.2f}')
        else:
            print(f'Grupo {grupo}: insuficiente para Mann-Whitney (n < 10).')

    modelo = ajustar_modelo_ols(df)
    return f_stat, p_val_anova, correlacoes, mann_whitney, modelo


def gerar_relatorio(
    df: pd.DataFrame,
    estat_por_faixa: pd.DataFrame,
    nota_por_regiao: pd.Series,
    elite: pd.DataFrame,
    vulneraveis: pd.DataFrame,
    p90: float,
    p10: float,
    modelo_ols: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    print('\n[Relatório] Salvando relatorio_final.md...')
    df_escola = df.groupby('TP_ESCOLA')['NOTA_MEDIA'].mean()
    diferenca_escola = df_escola.get(3, 0.0) - df_escola.get(2, 0.0)

    valor = lambda serie, chave: serie.get(chave, 0.0)
    escola_elite_privada = valor(elite['ESCOLA_DESCRICAO'].value_counts(normalize=True) * 100, 'Privada')
    escola_vuln_privada = valor(vulneraveis['ESCOLA_DESCRICAO'].value_counts(normalize=True) * 100, 'Privada')
    raca_vuln_preta = valor(vulneraveis['RACA_DESCRICAO'].value_counts(normalize=True) * 100, 'Preta')
    raca_vuln_parda = valor(vulneraveis['RACA_DESCRICAO'].value_counts(normalize=True) * 100, 'Parda')

    texto = f"""# Relatório Final: Renda Familiar e Desempenho no ENEM 2023

- Faixa A: média = {estat_por_faixa.loc['A', 'Media']:.1f}
- Faixa Q: média = {estat_por_faixa.loc['Q', 'Media']:.1f}
- Diferença Q-A: {estat_por_faixa.loc['Q', 'Media'] - estat_por_faixa.loc['A', 'Media']:.1f}
- Diferença entre escola privada e pública: {diferenca_escola:.1f}
- Elite P90+ com escola privada: {escola_elite_privada:.1f}%
- Vulneráveis P10- com escola privada: {escola_vuln_privada:.1f}%
- Pretos e pardos no P10-: {raca_vuln_preta + raca_vuln_parda:.1f}%
- Região mais alta: {nota_por_regiao.index[0]} ({nota_por_regiao.iloc[0]:.2f})
- Região mais baixa: {nota_por_regiao.index[-1]} ({nota_por_regiao.iloc[-1]:.2f})
- R² do modelo OLS: {modelo_ols.rsquared:.3f}
- Coeficiente renda (R$1k): {modelo_ols.params.get('RENDA_NUMERICA', 0.0) * 1000:.2f}
"""

    with open('relatorio_final.md', 'w', encoding='utf-8') as arquivo:
        arquivo.write(texto)
    print('[Relatório] relatorio_final.md salvo.')


def main() -> None:
    criar_pasta_saida(OUTPUT_DIR)
    df = carregar_dados(DATA_PATH)
    df = filtrar_dados(df)
    df = amostrar_dados(df)
    df = criar_variaveis_derivadas(df)
    imprimir_estatisticas_iniciais(df)
    estat_por_faixa = analise_por_faixa_renda(df)
    tabela_cruzada_renda_escola(df)
    nota_por_regiao = analise_uf_regiao(df)
    elite, vulneraveis, p90, p10 = perfil_percentis(df)
    gerar_visualizacoes(df)
    _, _, _, _, modelo_ols = executar_testes_estatisticos(df)
    gerar_relatorio(df, estat_por_faixa, nota_por_regiao, elite, vulneraveis, p90, p10, modelo_ols)
    print('\n--- Execução concluída. ---')


if __name__ == '__main__':
    main()
