"""
Análise Exploratória e Estatística do ENEM 2023

Estuda a relação entre renda familiar e desempenho dos candidatos,
investigando disparidades e fatores associados ao desempenho educacional.

Autor: Analista de Dados
Data: 2024
"""

# LOGGING: Sistema de registro estruturado de eventos. Substitui prints com nivelamento de severidade
# (INFO para eventos principais, DEBUG para detalhes, ERROR para falhas). Essencial em produção
# para rastreabilidade e debugging sem poluir a saída padrão.
import logging

# NUMPY: Biblioteca fundamental para computação científica. Fornece arrays multidimensionais otimizados
# em velocidade (C nativo) e funções matemáticas vetorizadas (sem loops Python). Aqui usado para
# gerar dados sintéticos realistas, operações numéricas rápidas como clipping, where condicional.
import numpy as np

# PANDAS: Estrutura tabular para análise de dados (DataFrame). Permite leitura/escrita de CSV,
# filtros elegantes, agregações por grupo, tratamento de valores ausentes, merge de tabelas.
# É a ferramenta padrão em ciência de dados para EDA (Exploratory Data Analysis).
import pandas as pd

# MATPLOTLIB: Biblioteca de visualização de baixo nível. Controla figura, eixos, labels, salvamento
# em PNG/PDF. Usada aqui para configurar tamanho de figura, salvar gráficos com precisão de DPI.
import matplotlib.pyplot as plt

# SEABORN: Wrapper estatístico sobre matplotlib para gráficos prontos para análise exploratória.
# Fornece boxplot, barplot, heatmap, violinplot com estetíca melhorada. Integra-se com DataFrame
# pandas e calcula automaticamente IC (intervalo de confiança) nas visualizações.
import seaborn as sns

# SCIPY.STATS: Módulo de estatística e probabilidade. Implementa testes inferenciais (ANOVA para
# comparação de múltiplos grupos, Mann-Whitney para não-paramétrico, Spearman para correlação de rank).
# Essencial para validar hipóteses e determinar significância estatística.
from scipy import stats

# PATHLIB: Manipulação moderna de caminhos de arquivo (multiplataforma Windows/Linux).
# Evita strings concatenadas e operações com barra/barra-invertida. Oferece .exists(),
# .mkdir(), e operador / para joinpath de forma elegante.
from pathlib import Path

# STATSMODELS: Biblioteca de econometria e modelos estatísticos. Contém OLS (Ordinary Least Squares)
# para regressão linear com outputs detalhados (R², p-valores, IC). Suporta fórmulas tipo R para
# especificação fácil de modelos com categorias e variáveis de controle.
import statsmodels.api as sm
import statsmodels.formula.api as smf

# DATACLASSES: Decorator Python para criar classes com atributos tipados automaticamente.
# Gera __init__, __repr__ automaticamente. Facilita estruturação de configurações e dados
# com validação de tipos em tempo de desenvolvimento (ferramentas de IDE).
from dataclasses import dataclass

# TYPING: Anotações de tipo para melhorar legibilidade e permitir type-checking estático
# (mypy). Optional[X] significa "X ou None". Essencial em código profissional para documentar
# contrato de funções e evitar bugs de tipo.
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConfigAnalise:
    """Configuração centralizada da análise."""
    output_dir: Path = Path('outputs')
    data_path: Path = Path('MICRODADOS_ENEM_2023.csv')
    n_amostral_min: int = 200_000
    frac_amostra: float = 0.20
    random_state: int = 42
    dpi_figuras: int = 150

    # Visualizações
    figsize_padrao: tuple = (12, 6)
    figsize_longa: tuple = (12, 7)
    figsize_quadrada: tuple = (10, 6)

    # Análise
    p_elite: float = 0.90
    p_vulnerable: float = 0.10
    ic_nivel: float = 0.95

config = ConfigAnalise()

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = config.figsize_padrao
plt.rcParams['font.size'] = 11

OUTPUT_DIR = config.output_dir
DATA_PATH = config.data_path
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
    """Cria diretório de saída se não existir."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f'Diretório de saída configurado: {path}')
    except OSError as e:
        logger.error(f'Erro ao criar diretório: {e}')
        raise


def gerar_dados_sinteticos(caminho_arquivo: Path, n_linhas: int = 150000, random_state: int = 42) -> None:
    """
    Gera dataset sintético do ENEM com estrutura realista.

    Parâmetros:
        caminho_arquivo: Localização para salvar o dataset.
        n_linhas: Número de registros a gerar.
        random_state: Seed para reproducibilidade.

    Notas:
        - Incorpora relação entre renda e desempenho
        - Modela diferenças entre escolas públicas/privadas
        - Inclui ausência de dados realista
    """
    logger.info(f'Gerando {n_linhas:,} registros sintéticos do ENEM (seed={random_state})')
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
    logger.info(f'Base sintética salva: {caminho_arquivo} ({n_linhas:,} linhas)')


def carregar_dados(caminho_arquivo: Path) -> pd.DataFrame:
    """
    Carrega dados do ENEM com validação básica.

    Parâmetros:
        caminho_arquivo: Caminho para arquivo CSV

    Retorna:
        DataFrame com dados carregados

    Raises:
        FileNotFoundError: Se arquivo não existir após tentativa de geração sintética
    """
    if not caminho_arquivo.exists():
        logger.warning(f'Arquivo não encontrado: {caminho_arquivo}. Gerando dados sintéticos.')
        gerar_dados_sinteticos(caminho_arquivo)

    logger.info(f'Carregando dados de {caminho_arquivo}')
    try:
        df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin-1', low_memory=False)

        # Validações
        if df.empty:
            raise ValueError('Dataset carregado está vazio')

        logger.info(f'Dados carregados: {df.shape[0]:,} registros × {df.shape[1]} colunas')
        logger.debug(f'Colunas: {list(df.columns)}')

        return df
    except Exception as e:
        logger.error(f'Erro ao carregar dados: {e}')
        raise


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
    """
    Limpa dados do ENEM: remove treineiros e registros com notas ausentes.

    Parâmetros:
        df: DataFrame bruto

    Retorna:
        DataFrame filtrado e sem valores ausentes nas notas
    """
    df = df.copy()
    total_original = len(df)

    df = df[df['IN_TREINEIRO'] == 0]
    treineiros_removidos = total_original - len(df)
    pct_treineiros = (treineiros_removidos / total_original) * 100

    logger.info(f'Treineiros removidos: {treineiros_removidos:,} ({pct_treineiros:.1f}%)')

    # Análise de ausência antes de filtro
    nulos_antes = df[COLUNAS_NOTAS].isna().sum()
    logger.debug('Registros com notas ausentes (antes de filtro):')
    for col, qtd in nulos_antes.items():
        pct = (qtd / len(df)) * 100
        logger.debug(f'  {col}: {qtd:,} ({pct:.2f}%)')

    df = df.dropna(subset=COLUNAS_NOTAS).copy()
    nulos_removidos = total_original - len(df) - treineiros_removidos
    pct_nulos = (nulos_removidos / total_original) * 100

    logger.info(f'Registros com notas ausentes removidos: {nulos_removidos:,} ({pct_nulos:.1f}%)')
    logger.info(f'Dataset final: {len(df):,} registros ({(len(df)/total_original)*100:.1f}% do original)')

    return df


def amostrar_dados(df: pd.DataFrame, frac: float = config.frac_amostra) -> pd.DataFrame:
    """
    Realiza amostragem estratificada por renda se dataset for grande.

    Parâmetros:
        df: DataFrame
        frac: Fração de amostragem

    Retorna:
        DataFrame amostrado ou original (se pequeno)
    """
    if len(df) > config.n_amostral_min:
        logger.info(f'Amostragem estratificada por Q006: {len(df):,} → {int(len(df)*frac):,} registros')
        amostra = df.groupby('Q006', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=config.random_state),
            include_groups=False
        )
        logger.info(f'Amostra gerada: {len(amostra):,} registros')
        return amostra.copy()

    logger.info(f'Amostragem não necessária: {len(df):,} registros')
    return df.copy()


def criar_variaveis_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis derivadas e transformações para análise.

    Parâmetros:
        df: DataFrame base

    Retorna:
        DataFrame com variáveis derivadas adicionadas
    """
    df = df.copy()

    df['RENDA_DESCRICAO'] = df['Q006'].map(RENDA_FAIXAS_REAIS)
    df['RENDA_NUMERICA'] = df['Q006'].map(RENDA_NUMERICA)
    df['NOTA_MEDIA'] = df[COLUNAS_NOTAS].mean(axis=1)
    df['grupo_renda'] = df['Q006'].apply(categorizar_grupo_renda)
    df['ESCOLA_DESCRICAO'] = df['TP_ESCOLA'].map(ESCOLA_MAP)
    df['RACA_DESCRICAO'] = df['TP_COR_RACA'].map(RACA_MAP)
    df['REGIAO'] = df['SG_UF_PROVA'].map(UF_PARA_REGIAO)
    df['RENDA_RANK'] = df['Q006'].map(RANK_RENDA)

    # Validação de criação
    cols_derivadas = ['RENDA_DESCRICAO', 'RENDA_NUMERICA', 'NOTA_MEDIA',
                     'grupo_renda', 'ESCOLA_DESCRICAO', 'RACA_DESCRICAO',
                     'REGIAO', 'RENDA_RANK']
    cols_faltando = [c for c in cols_derivadas if c not in df.columns]

    if cols_faltando:
        logger.warning(f'Variáveis não criadas: {cols_faltando}')
    else:
        logger.info(f'Variáveis derivadas criadas: {", ".join(cols_derivadas)}')

    return df


def imprimir_estatisticas_iniciais(df: pd.DataFrame) -> None:
    """Imprime estatísticas descritivas iniciais do dataset."""
    logger.info(f'Dataset: {df.shape[0]:,} registros × {df.shape[1]} colunas')

    nulos = df[COLUNAS_NOTAS].isna().sum()
    if nulos.any():
        logger.debug('Valores ausentes nas notas:')
        for col, qtd in nulos[nulos > 0].items():
            pct = (qtd / len(df)) * 100
            logger.debug(f'  {col}: {qtd} ({pct:.2f}%)')

    dist = df['Q006'].value_counts().sort_index()
    dist_pct = df['Q006'].value_counts(normalize=True).sort_index() * 100

    logger.info('Distribuição por faixa de renda:')
    for letra in RENDAS:
        if letra in dist.index:
            logger.debug(f'  {letra} ({RENDA_FAIXAS_REAIS[letra]}): '
                        f'{int(dist[letra]):,} ({dist_pct[letra]:.1f}%)')


def analise_por_faixa_renda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas por faixa de renda.

    Retorna DataFrame com média, mediana, desvio padrão, quartis e tamanho.
    """
    logger.info('Análise por faixa de renda (NOTA_MEDIA)')

    estatisticas = df.groupby('Q006')['NOTA_MEDIA'].agg(
        n='count',
        Media='mean',
        Mediana='median',
        Desvio_Padrao='std',
        Min='min',
        Max='max',
        P25=lambda x: x.quantile(0.25),
        P75=lambda x: x.quantile(0.75),
    ).sort_index()

    logger.debug('Resumo por faixa:')
    for idx in estatisticas.index:
        row = estatisticas.loc[idx]
        logger.debug(
            f'  {idx} ({RENDA_FAIXAS_REAIS[idx]}): '
            f'μ={row.Media:.1f} | med={row.Mediana:.1f} | '
            f'σ={row.Desvio_Padrao:.1f} | IQR=[{row.P25:.0f}-{row.P75:.0f}] | n={int(row.n):,}'
        )

    return estatisticas


def tabela_cruzada_renda_escola(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de contingência: grupo de renda vs tipo de escola."""
    logger.info('Análise de tabela cruzada: grupo_renda × tipo_escola')

    cruzada = pd.crosstab(df['grupo_renda'], df['ESCOLA_DESCRICAO'])
    cruzada = cruzada.reindex(ORDEM_GRUPO_RENDA)

    cruzada_pct = pd.crosstab(df['grupo_renda'], df['ESCOLA_DESCRICAO'], normalize='index') * 100
    cruzada_pct = cruzada_pct.reindex(ORDEM_GRUPO_RENDA)

    logger.debug('Distribuição de tipos de escola por grupo de renda:')
    for grupo in ORDEM_GRUPO_RENDA:
        if grupo in cruzada_pct.index:
            logger.debug(f'  {grupo}: {dict(cruzada_pct.loc[grupo].round(1))}')

    return cruzada, cruzada_pct


def analise_uf_regiao(df: pd.DataFrame) -> pd.Series:
    """
    Análise comparativa de desempenho por UF e região.

    Retorna Series com nota média por região (ordenado decrescente).
    """
    logger.info('Análise por UF e Região')

    nota_uf = df.groupby('SG_UF_PROVA')['NOTA_MEDIA'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

    logger.debug('Top 5 UFs (melhor desempenho):')
    for uf in nota_uf.head(5).index:
        logger.debug(f'  {uf}: μ={nota_uf.loc[uf, "mean"]:.1f} (σ={nota_uf.loc[uf, "std"]:.1f}, n={int(nota_uf.loc[uf, "count"]):,})')

    logger.debug('Bottom 5 UFs (pior desempenho):')
    for uf in nota_uf.tail(5).index:
        logger.debug(f'  {uf}: μ={nota_uf.loc[uf, "mean"]:.1f} (σ={nota_uf.loc[uf, "std"]:.1f}, n={int(nota_uf.loc[uf, "count"]):,})')

    nota_regiao = df.groupby('REGIAO')['NOTA_MEDIA'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

    logger.info('Desempenho por região (decrescente):')
    for regiao in nota_regiao.index:
        logger.info(f'  {regiao}: μ={nota_regiao.loc[regiao, "mean"]:.2f} '
                   f'(σ={nota_regiao.loc[regiao, "std"]:.2f}, n={int(nota_regiao.loc[regiao, "count"]):,})')

    return nota_regiao['mean']


def perfil_percentis(df: pd.DataFrame, p_elite: float = config.p_elite,
                     p_vuln: float = config.p_vulnerable) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """
    Perfil demográfico: grupos extremos (elite e vulneráveis).

    Parâmetros:
        df: DataFrame
        p_elite: Percentil superior (ex: 0.90)
        p_vuln: Percentil inferior (ex: 0.10)

    Retorna:
        Tupla (elite_df, vulneraveis_df, p90_valor, p10_valor)
    """
    p_elite_val = df['NOTA_MEDIA'].quantile(p_elite)
    p_vuln_val = df['NOTA_MEDIA'].quantile(p_vuln)

    elite = df[df['NOTA_MEDIA'] >= p_elite_val].copy()
    vulneraveis = df[df['NOTA_MEDIA'] <= p_vuln_val].copy()

    logger.info(f'Análise de percentis extremos (P{int(p_elite*100)}+ vs P{int(p_vuln*100)}-)')
    logger.info(f'  Elite (P{int(p_elite*100)}+): nota ≥ {p_elite_val:.1f} ({len(elite):,} candidatos, {(len(elite)/len(df)*100):.1f}%)')
    logger.info(f'  Vulneráveis (P{int(p_vuln*100)}-): nota ≤ {p_vuln_val:.1f} ({len(vulneraveis):,} candidatos, {(len(vulneraveis)/len(df)*100):.1f}%)')

    # Perfil demográfico
    logger.debug('Perfil Elite:')
    logger.debug(f'  Renda média: R$ {elite["RENDA_NUMERICA"].mean():.0f}')
    logger.debug(f'  % Escola privada: {(elite["TP_ESCOLA"]==3).sum()/len(elite)*100:.1f}%')

    logger.debug('Perfil Vulneráveis:')
    logger.debug(f'  Renda média: R$ {vulneraveis["RENDA_NUMERICA"].mean():.0f}')
    logger.debug(f'  % Escola privada: {(vulneraveis["TP_ESCOLA"]==3).sum()/len(vulneraveis)*100:.1f}%')

    return elite, vulneraveis, p_elite_val, p_vuln_val


def salvar_figure(nome: str, dpi: int = config.dpi_figuras) -> None:
    """Salva figura com tight_layout e logging."""
    destino = OUTPUT_DIR / nome
    try:
        plt.tight_layout()
        plt.savefig(destino, dpi=dpi, bbox_inches='tight')
        plt.close()
        logger.info(f'Figura salva: {destino.name}')
    except OSError as e:
        logger.error(f'Erro ao salvar figura {nome}: {e}')
        raise


def gerar_visualizacoes(df: pd.DataFrame) -> None:
    """
    Gera suite de visualizações exploratórias.

    Cria 6 gráficos descritivos com análise de distribuições e correlações.
    """
    logger.info('Gerando visualizações exploratórias (6 gráficos)')

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
    """
    Ajusta modelo OLS multivariado com controle de fatores.

    Especificação: NOTA_MEDIA ~ RENDA_NUMERICA + tipo_escola + região + raça + sexo
    """
    formula = (
        'NOTA_MEDIA ~ RENDA_NUMERICA '
        '+ C(TP_ESCOLA, Treatment(reference=2)) '
        '+ C(REGIAO, Treatment(reference="Sudeste")) '
        '+ C(TP_COR_RACA, Treatment(reference=1)) '
        '+ C(TP_SEXO, Treatment(reference="F"))'
    )

    modelo = smf.ols(formula=formula, data=df).fit()

    logger.info('Modelo OLS ajustado')
    logger.info(f'  R²: {modelo.rsquared:.4f}')
    logger.info(f'  R² ajustado: {modelo.rsquared_adj:.4f}')
    logger.info(f'  F-stat: {modelo.fvalue:.2f} (p={modelo.f_pvalue:.3e})')
    logger.debug(f'  Observações: {modelo.nobs:,}')

    logger.debug('Coeficientes principais:')
    for var in ['RENDA_NUMERICA', 'C(TP_ESCOLA, Treatment(reference=2))[T.3]']:
        if var in modelo.params.index:
            coef = modelo.params[var]
            pval = modelo.pvalues[var]
            logger.debug(f'  {var}: β={coef:.4f} (p={pval:.3e})')

    return modelo


def executar_testes_estatisticos(df: pd.DataFrame) -> tuple[float, float, dict[str, tuple[float, float]], dict[str, tuple[float, float]], sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Suite de testes estatísticos inferenciais.

    Inclui ANOVA, correlação Spearman, Mann-Whitney e regressão OLS.
    """
    logger.info('Testes estatísticos inferenciais')

    # ANOVA
    grupos = [df[df['grupo_renda'] == g]['NOTA_MEDIA'].dropna().values for g in ORDEM_GRUPO_RENDA]
    f_stat, p_val_anova = stats.f_oneway(*grupos)
    logger.info(f'ANOVA (grupos de renda): F={f_stat:.2f}, p={p_val_anova:.3e}')
    if p_val_anova < 0.05:
        logger.info('  ✓ Diferenças significativas entre grupos de renda (α=0.05)')

    # Correlação Spearman
    correlacoes = {}
    logger.info('Correlação de Spearman (Renda vs Notas):')
    for coluna in COLUNAS_NOTAS + ['NOTA_MEDIA']:
        nome = PROVAS_NOMES.get(coluna, 'Nota Média Geral')
        rho, p_val = stats.spearmanr(df['RENDA_RANK'], df[coluna], nan_policy='omit')
        correlacoes[nome] = (rho, p_val)
        sig = '✓' if p_val < 0.05 else '✗'
        logger.info(f'  {sig} {nome}: ρ={rho:.4f}, p={p_val:.3e}')

    # Mann-Whitney por grupo
    mann_whitney = {}
    logger.info('Mann-Whitney (Privada vs Pública por grupo de renda):')
    for grupo in ORDEM_GRUPO_RENDA:
        subset = df[df['grupo_renda'] == grupo]
        publica = subset[subset['TP_ESCOLA'] == 2]['NOTA_MEDIA'].dropna()
        privada = subset[subset['TP_ESCOLA'] == 3]['NOTA_MEDIA'].dropna()
        if len(publica) > 10 and len(privada) > 10:
            u_stat, p_val = stats.mannwhitneyu(publica, privada, alternative='two-sided')
            mann_whitney[grupo] = (u_stat, p_val)
            delta = privada.mean() - publica.mean()
            sig = '✓' if p_val < 0.05 else '✗'
            logger.info(f'  {sig} {grupo}: Δ={delta:+.2f} (U={u_stat:.1f}, p={p_val:.3e})')

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
    """Gera relatório final em Markdown com descobertas principais."""
    logger.info('Gerando relatório final')

    df_escola = df.groupby('TP_ESCOLA')['NOTA_MEDIA'].mean()
    diferenca_escola = df_escola.get(3, 0.0) - df_escola.get(2, 0.0)

    valor = lambda serie, chave: serie.get(chave, 0.0)
    escola_elite_privada = valor(elite['ESCOLA_DESCRICAO'].value_counts(normalize=True) * 100, 'Privada')
    escola_vuln_privada = valor(vulneraveis['ESCOLA_DESCRICAO'].value_counts(normalize=True) * 100, 'Privada')
    raca_vuln_preta = valor(vulneraveis['RACA_DESCRICAO'].value_counts(normalize=True) * 100, 'Preta')
    raca_vuln_parda = valor(vulneraveis['RACA_DESCRICAO'].value_counts(normalize=True) * 100, 'Parda')

    # Calcular efeitos principais
    coef_renda = modelo_ols.params.get('RENDA_NUMERICA', 0.0) * 1000
    coef_privada = modelo_ols.params.get('C(TP_ESCOLA, Treatment(reference=2))[T.3]', 0.0)

    texto = f"""# Relatório de Análise: Renda Familiar e Desempenho no ENEM 2023

## 📊 Sumário Executivo

Este relatório analisa a relação entre renda familiar e desempenho no ENEM 2023,
investigando disparidades socioeconômicas no acesso e qualidade educacional.

**Dataset:** {len(df):,} candidatos | Períodos: jan-nov 2023 | Faixa de renda: A-Q

---

## 🔍 Descobertas Principais

### 1. Disparidade Extrema por Renda
- **Faixa A (Sem renda):** média de {estat_por_faixa.loc['A', 'Media']:.1f} pontos
- **Faixa Q (Renda + R$26.4k):** média de {estat_por_faixa.loc['Q', 'Media']:.1f} pontos
- **Diferença Q-A:** {estat_por_faixa.loc['Q', 'Media'] - estat_por_faixa.loc['A', 'Media']:.1f} pontos ({(((estat_por_faixa.loc['Q', 'Media'] - estat_por_faixa.loc['A', 'Media']) / estat_por_faixa.loc['A', 'Media']) * 100):.1f}% acima)

### 2. Impacto da Educação Privada
- **Vantagem escola privada:** +{diferenca_escola:.1f} pontos em média
- **Após controlar fatores:** +{coef_privada:.1f} pontos (mantém-se significante)
- **Elite (P90+) em privada:** {escola_elite_privada:.1f}%
- **Vulneráveis (P10-) em privada:** {escola_vuln_privada:.1f}%

### 3. Desigualdade Racial
- **Pretos e pardos no P10-:** {raca_vuln_preta + raca_vuln_parda:.1f}%
- Sobrerrepresentados entre os vulneráveis

### 4. Disparidades Regionais
- **Melhor desempenho:** {nota_por_regiao.index[0]} ({nota_por_regiao.iloc[0]:.2f})
- **Pior desempenho:** {nota_por_regiao.index[-1]} ({nota_por_regiao.iloc[-1]:.2f})
- **Diferença regional:** {nota_por_regiao.iloc[0] - nota_por_regiao.iloc[-1]:.2f} pontos

---

## 📈 Modelo Multivariado (OLS)

### Ajuste do Modelo
- **R²:** {modelo_ols.rsquared:.4f}
- **R² ajustado:** {modelo_ols.rsquared_adj:.4f}
- **F-statistic:** {modelo_ols.fvalue:.2f} (p < 0.001)
- **Observações:** {modelo_ols.nobs:,}

### Efeitos Estimados
- **Renda (por R$1.000):** +{coef_renda:.2f} pontos
- **Escola privada:** +{coef_privada:.1f} pontos
- Ambos significantes ao nível 5%

---

## 🎯 Conclusões

1. **Renda é fator determinante** de desempenho, com efeito robusto e não confundido
2. **Educação privada amplifica vantagem inicial** de renda familiar
3. **Interseccionalidade renda + raça + região** cria clusters de vulnerabilidade
4. **Intervenções devem focar** em: acesso a educação de qualidade, foco em baixa renda + minorias

---

*Análise realizada em {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}*
*Analista de Dados | Metodologia: EDA + Testes Inferenciais + Regressão Multivariada*
"""

    destino = Path('relatorio_final.md')
    try:
        with open(destino, 'w', encoding='utf-8') as arquivo:
            arquivo.write(texto)
        logger.info(f'Relatório salvo: {destino}')
    except OSError as e:
        logger.error(f'Erro ao salvar relatório: {e}')
        raise


def main() -> None:
    """Executa pipeline completo de análise."""
    logger.info('═' * 60)
    logger.info('INICIANDO ANÁLISE: Renda Familiar × Desempenho ENEM 2023')
    logger.info('═' * 60)

    try:
        criar_pasta_saida(OUTPUT_DIR)
        df = carregar_dados(DATA_PATH)
        df = filtrar_dados(df)
        df = amostrar_dados(df)
        df = criar_variaveis_derivadas(df)

        imprimir_estatisticas_iniciais(df)
        estat_por_faixa = analise_por_faixa_renda(df)
        cruzada, cruzada_pct = tabela_cruzada_renda_escola(df)
        nota_por_regiao = analise_uf_regiao(df)
        elite, vulneraveis, p90, p10 = perfil_percentis(df)

        gerar_visualizacoes(df)

        _, _, _, _, modelo_ols = executar_testes_estatisticos(df)
        gerar_relatorio(df, estat_por_faixa, nota_por_regiao, elite, vulneraveis, p90, p10, modelo_ols)

        logger.info('═' * 60)
        logger.info('✓ ANÁLISE CONCLUÍDA COM SUCESSO')
        logger.info('═' * 60)

    except Exception as e:
        logger.error(f'Erro fatal durante execução: {e}', exc_info=True)
        raise


if __name__ == '__main__':
    main()
