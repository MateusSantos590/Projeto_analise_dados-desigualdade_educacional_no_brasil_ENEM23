"""
Análise Exploratória e Estatística do ENEM 2023

Estuda a relação entre renda familiar e desempenho dos candidatos,
investigando disparidades e fatores associados ao desempenho educacional.

Autor: Mateus Santos Telles
Curso: Análise e Desenvolvimento de Sistemas - UNG
Disciplina: Programação Orientada a Objetos aplicada à Ciência de Dados
Data: 2024

Arquitetura: O projeto foi estruturado em 5 classes principais para demonstrar
a aplicação dos 4 pilares da POO (Abstração, Encapsulamento, Herança e Polimorfismo)
em um pipeline real de análise de dados.
"""
"Cada parte deste código documentei passo a passo para melhor compreesão."

# =============================================================================
# IMPORTAÇÕES E CONFIGURAÇÃO
# =============================================================================

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTES GLOBAIS (Encapsuladas em dataclasses ou dicionários)
# =============================================================================

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
    'A': 0.0, 'B': 660.0, 'C': 1650.0, 'D': 2310.0, 'E': 2970.0,
    'F': 3630.0, 'G': 4620.0, 'H': 5940.0, 'I': 7260.0, 'J': 8580.0,
    'K': 9900.0, 'L': 11220.0, 'M': 12540.0, 'N': 14520.0, 'O': 17820.0,
    'P': 23100.0, 'Q': 30000.0,
}

ORDEM_GRUPO_RENDA = ['Baixa', 'Média-baixa', 'Média', 'Média-alta', 'Alta']
COLUNAS_NOTAS = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
PROVAS_NOMES = {
    'NU_NOTA_CN': 'Ciências Natureza', 'NU_NOTA_CH': 'Ciências Humanas',
    'NU_NOTA_LC': 'Linguagens', 'NU_NOTA_MT': 'Matemática',
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

# =============================================================================
# CLASSE 1: CONFIGURAÇÃO (Dataclass + Encapsulamento)
# Demonstra: Uso de dataclasses para organizar parâmetros
# =============================================================================

@dataclass
class ConfigAnalise:
    """
    Configuração centralizada da análise.
    Encapsula todos os parâmetros ajustáveis em um único objeto imutável.
    """
    output_dir: Path = Path('outputs')
    data_path: Path = Path('MICRODADOS_ENEM_2023.csv')
    n_amostral_min: int = 200_000
    frac_amostra: float = 0.20
    random_state: int = 42
    dpi_figuras: int = 150

    figsize_padrao: tuple = (12, 6)
    figsize_longa: tuple = (12, 7)
    figsize_quadrada: tuple = (10, 6)

    p_elite: float = 0.90
    p_vulnerable: float = 0.10
    ic_nivel: float = 0.95

# =============================================================================
# CLASSE 2: CLASSE ABSTRATA PARA ANÁLISE (Abstração + Polimorfismo)
# Demonstra: Uso de ABC para definir interface de pipeline
# =============================================================================

class AnaliseENEM(ABC):
    """
    Classe abstrata que define o contrato para qualquer análise do ENEM.
    Demonstra o pilar da ABSTRAÇÃO: define O QUE deve ser feito, não COMO.
    """

    def __init__(self, config: ConfigAnalise) -> None:
        self.config = config
        self.df: Optional[pd.DataFrame] = None

    @abstractmethod
    def carregar_dados(self) -> pd.DataFrame:
        """Carrega os dados brutos do ENEM."""
        pass

    @abstractmethod
    def limpar_dados(self) -> pd.DataFrame:
        """Aplica filtros e limpeza nos dados."""
        pass

    @abstractmethod
    def criar_variaveis(self) -> pd.DataFrame:
        """Cria variáveis derivadas para análise."""
        pass

    @abstractmethod
    def executar_analise(self) -> None:
        """Executa o pipeline completo de análise."""
        pass

# =============================================================================
# CLASSE 3: GERADOR DE DADOS SINTÉTICOS (Responsabilidade única)
# Demonstra: Princípio da Responsabilidade Única (SOLID)
# =============================================================================

class GeradorDadosSinteticos:
    """
    Classe dedicada exclusivamente à geração de dados sintéticos do ENEM.
    Demonstra o princípio S (Single Responsibility) do SOLID.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        np.random.seed(random_state)

    def gerar(self, n_linhas: int = 150000) -> pd.DataFrame:
        """
        Gera dataset sintético do ENEM com estrutura realista.

        Returns:
            DataFrame com dados sintéticos prontos para análise.
        """
        logger.info(f'Gerando {n_linhas:,} registros sintéticos do ENEM')

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

        renda_num = np.array([RENDA_NUMERICA[letra] for letra in q006_array])
        prob_internet = (
            0.18 + 0.00006 * renda_num +
            0.10 * (tp_escola == 3) +
            0.08 * np.isin(sg_ufs, ['SP', 'RJ', 'MG'])
        )
        prob_internet = np.clip(prob_internet, 0.05, 0.95)
        internet_banda_larga = (np.random.random(n_linhas) < prob_internet).astype(int)

        renda_efeito = renda_num * 15.0
        escola_efeito = np.where(tp_escola == 3, 60.0, 0.0) + np.where(tp_escola == 2, -15.0, 0.0)
        nota_base = 450.0 + renda_efeito + escola_efeito

        nu_nota_cn = np.clip(np.random.normal(nota_base * 0.90, 65.0), 0, 1000)
        nu_nota_ch = np.clip(np.random.normal(nota_base * 1.05, 60.0), 0, 1000)
        nu_nota_lc = np.clip(np.random.normal(nota_base * 1.02, 55.0), 0, 1000)
        nu_nota_mt = np.clip(np.random.normal(nota_base * 0.95 + 15.0 + 18.0 * internet_banda_larga, 80.0), 0, 1000)
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
            'INTERNET_BANDA_LARGA': internet_banda_larga,
        })

        logger.info(f'Base sintética gerada: {n_linhas:,} registros')
        return df_sint

    def salvar(self, df: pd.DataFrame, caminho: Path) -> None:
        """Salva o dataset sintético em CSV."""
        df.to_csv(caminho, sep=';', index=False, encoding='latin-1')
        logger.info(f'Base sintética salva: {caminho}')

# =============================================================================
# CLASSE 4: PIPELINE PRINCIPAL (Herança de AnaliseENEM)
# Demonstra: HERANÇA da classe abstrata e implementação concreta
# =============================================================================

class PipelineAnaliseENEM(AnaliseENEM):
    """
    Implementação concreta do pipeline de análise do ENEM.
    Herda de AnaliseENEM e implementa todos os métodos abstratos.
    Demonstra o pilar da HERANÇA.
    """

    def __init__(self, config: ConfigAnalise) -> None:
        super().__init__(config)
        self.gerador_sintetico = GeradorDadosSinteticos(config.random_state)
        # Configuração visual
        sns.set_theme(style='whitegrid')
        plt.rcParams['figure.figsize'] = config.figsize_padrao
        plt.rcParams['font.size'] = 11

    def _criar_pasta_saida(self) -> None:
        """Método privado: encapsulamento da lógica de criação de diretório."""
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f'Diretório configurado: {self.config.output_dir}')
        except OSError as e:
            logger.error(f'Erro ao criar diretório: {e}')
            raise

    def carregar_dados(self) -> pd.DataFrame:
        """
        Carrega dados do ENEM. Se não existir, gera dados sintéticos.
        POLIMORFISMO: implementação concreta do método abstrato.
        """
        if not self.config.data_path.exists():
            logger.warning(f'Arquivo não encontrado. Gerando dados sintéticos...')
            df_sint = self.gerador_sintetico.gerar()
            self.gerador_sintetico.salvar(df_sint, self.config.data_path)

        logger.info(f'Carregando dados de {self.config.data_path}')
        try:
            df = pd.read_csv(self.config.data_path, sep=';', encoding='latin-1', low_memory=False)
            if df.empty:
                raise ValueError('Dataset carregado está vazio')
            logger.info(f'Dados carregados: {df.shape[0]:,} registros × {df.shape[1]} colunas')
            self.df = df
            return df
        except Exception as e:
            logger.error(f'Erro ao carregar dados: {e}')
            raise

    def limpar_dados(self) -> pd.DataFrame:
        """
        Limpa dados: remove treineiros e registros com notas ausentes.
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute carregar_dados() primeiro.")

        df = self.df.copy()
        total_original = len(df)

        # Remove treineiros
        df = df[df['IN_TREINEIRO'] == 0]
        treineiros_removidos = total_original - len(df)
        logger.info(f'Treineiros removidos: {treineiros_removidos:,}')

        # Remove registros sem notas
        df = df.dropna(subset=COLUNAS_NOTAS).copy()
        logger.info(f'Dataset limpo: {len(df):,} registros')

        self.df = df
        return df

    def _amostrar_dados(self) -> pd.DataFrame:
        """
        Método privado: amostragem estratificada se dataset for grande.
        """
        if self.df is None:
            raise ValueError("Dados não carregados.")

        if len(self.df) > self.config.n_amostral_min:
            logger.info(f'Amostrando {len(self.df):,} → {int(len(self.df) * self.config.frac_amostra):,} registros')
            amostra = self.df.groupby('Q006', group_keys=False).apply(
                lambda x: x.sample(frac=self.config.frac_amostra, random_state=self.config.random_state),
                include_groups=False
            )
            self.df = amostra.copy()
        return self.df

    def criar_variaveis(self) -> pd.DataFrame:
        """
        Cria variáveis derivadas para análise.
        """
        if self.df is None:
            raise ValueError("Dados não carregados.")

        df = self.df.copy()

        # Mapeamentos básicos
        df['RENDA_DESCRICAO'] = df['Q006'].map(RENDA_FAIXAS_REAIS)
        df['RENDA_NUMERICA'] = df['Q006'].map(RENDA_NUMERICA)
        df['NOTA_MEDIA'] = df[COLUNAS_NOTAS].mean(axis=1)
        df['grupo_renda'] = df['Q006'].apply(self._categorizar_grupo_renda)
        df['ESCOLA_DESCRICAO'] = df['TP_ESCOLA'].map(ESCOLA_MAP)
        df['RACA_DESCRICAO'] = df['TP_COR_RACA'].map(RACA_MAP)
        df['REGIAO'] = df['SG_UF_PROVA'].map(UF_PARA_REGIAO)
        df['RENDA_RANK'] = df['Q006'].map(RANK_RENDA)

        # Proxy de internet (se não existir)
        if 'INTERNET_BANDA_LARGA' not in df.columns:
            df['INTERNET_BANDA_LARGA'] = self._criar_proxy_internet(df)

        df['INTERNET_BANDA_LARGA_DESCRICAO'] = np.where(
            df['INTERNET_BANDA_LARGA'] == 1, 'Com banda larga', 'Sem banda larga'
        )

        logger.info(f'Variáveis derivadas criadas com sucesso')
        self.df = df
        return df

    @staticmethod
    def _categorizar_grupo_renda(valor: str) -> str:
        """Método estático: não depende da instância."""
        if valor in ['A', 'B', 'C']: return 'Baixa'
        if valor in ['D', 'E', 'F']: return 'Média-baixa'
        if valor in ['G', 'H', 'I']: return 'Média'
        if valor in ['J', 'K', 'L', 'M']: return 'Média-alta'
        if valor in ['N', 'O', 'P', 'Q']: return 'Alta'
        return 'Indefinido'

    @staticmethod
    def _criar_proxy_internet(df: pd.DataFrame) -> pd.Series:
        """Cria proxy pedagógico de acesso à internet."""
        renda_num = df['RENDA_NUMERICA'].fillna(df['RENDA_NUMERICA'].median())
        escola_privada = (df['TP_ESCOLA'] == 3).astype(int)
        regiao_servico = df['SG_UF_PROVA'].map(UF_PARA_REGIAO).isin(['Sudeste', 'Sul', 'Centro-Oeste']).astype(int)

        prob = 1 / (1 + np.exp(-(-1.8 + 0.00006 * renda_num + 0.95 * escola_privada + 0.25 * regiao_servico)))
        limiar = prob.quantile(0.55)
        return (prob >= limiar).astype(int)

    def executar_analise(self) -> None:
        """
        Pipeline completo: carrega → limpa → amostra → cria variáveis → analisa.
        """
        logger.info('═' * 60)
        logger.info('INICIANDO PIPELINE DE ANÁLISE DO ENEM 2023')
        logger.info('═' * 60)

        try:
            self._criar_pasta_saida()
            self.carregar_dados()
            self.limpar_dados()
            self._amostrar_dados()
            self.criar_variaveis()

            # Instancia as classes de análise e visualização
            analisador = AnalisadorEstatistico(self.df, self.config)
            analisador.executar_todos_testes()

            visualizador = VisualizadorDados(self.df, self.config)
            visualizador.gerar_todos_graficos()

            relator = GeradorRelatorio(self.df, self.config)
            relator.gerar()

            logger.info('═' * 60)
            logger.info('✓ PIPELINE CONCLUÍDO COM SUCESSO')
            logger.info('═' * 60)

        except Exception as e:
            logger.error(f'Erro fatal: {e}', exc_info=True)
            raise

# =============================================================================
# CLASSE 5: ANALISADOR ESTATÍSTICO (Composição)
# Demonstra: COMPOSIÇÃO - PipelineAnaliseENEM "tem um" AnalisadorEstatistico
# =============================================================================

class AnalisadorEstatistico:
    """
    Classe responsável por todos os testes estatísticos.
    Demonstra o princípio da Composição sobre Herança.
    """

    def __init__(self, df: pd.DataFrame, config: ConfigAnalise) -> None:
        self.df = df
        self.config = config

    def anova_grupos_renda(self) -> tuple[float, float]:
        """ANOVA para comparar notas entre grupos de renda."""
        grupos = [self.df[self.df['grupo_renda'] == g]['NOTA_MEDIA'].dropna().values
                  for g in ORDEM_GRUPO_RENDA]
        f_stat, p_val = stats.f_oneway(*grupos)
        logger.info(f'ANOVA: F={f_stat:.2f}, p={p_val:.3e}')
        return f_stat, p_val

    def correlacao_spearman(self) -> dict[str, tuple[float, float]]:
        """Correlação de Spearman entre renda e notas."""
        correlacoes = {}
        for coluna in COLUNAS_NOTAS + ['NOTA_MEDIA']:
            nome = PROVAS_NOMES.get(coluna, 'Nota Média Geral')
            rho, p_val = stats.spearmanr(self.df['RENDA_RANK'], self.df[coluna], nan_policy='omit')
            correlacoes[nome] = (rho, p_val)
            logger.info(f'Spearman {nome}: ρ={rho:.4f}, p={p_val:.3e}')
        return correlacoes

    def modelo_ols_principal(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Modelo OLS multivariado principal."""
        formula = (
            'NOTA_MEDIA ~ RENDA_NUMERICA '
            '+ C(TP_ESCOLA, Treatment(reference=2)) '
            '+ C(REGIAO, Treatment(reference="Sudeste")) '
            '+ C(TP_COR_RACA, Treatment(reference=1)) '
            '+ C(TP_SEXO, Treatment(reference="F"))'
        )
        modelo = smf.ols(formula=formula, data=self.df).fit()
        logger.info(f'OLS Principal: R²={modelo.rsquared:.4f}')
        return modelo

    def modelo_internet_matematica(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        """Modelo focado no efeito da internet sobre matemática."""
        formula = (
            'NU_NOTA_MT ~ RENDA_NUMERICA + INTERNET_BANDA_LARGA '
            '+ C(TP_ESCOLA, Treatment(reference=2)) '
            '+ C(REGIAO, Treatment(reference="Sudeste")) '
            '+ C(TP_COR_RACA, Treatment(reference=1)) '
            '+ C(TP_SEXO, Treatment(reference="F"))'
        )
        modelo = smf.ols(formula=formula, data=self.df).fit()
        logger.info(f'OLS Matemática: R²={modelo.rsquared:.4f}')
        return modelo

    def executar_todos_testes(self) -> dict[str, Any]:
        """Executa todos os testes e retorna resultados."""
        logger.info('Executando bateria de testes estatísticos...')
        resultados = {
            'anova': self.anova_grupos_renda(),
            'spearman': self.correlacao_spearman(),
            'modelo_ols': self.modelo_ols_principal(),
            'modelo_internet': self.modelo_internet_matematica(),
        }
        return resultados

# =============================================================================
# CLASSE 6: VISUALIZADOR (Responsabilidade única para gráficos)
# =============================================================================

class VisualizadorDados:
    """
    Classe dedicada exclusivamente à geração de visualizações.
    """

    def __init__(self, df: pd.DataFrame, config: ConfigAnalise) -> None:
        self.df = df
        self.config = config

    def _salvar_figura(self, nome: str) -> None:
        """Método privado para salvar figura com padronização."""
        destino = self.config.output_dir / nome
        plt.tight_layout()
        plt.savefig(destino, dpi=self.config.dpi_figuras, bbox_inches='tight')
        plt.close()
        logger.info(f'Figura salva: {destino.name}')

    def boxplot_renda(self) -> None:
        """Boxplot da nota média por grupo de renda."""
        plt.figure(figsize=self.config.figsize_quadrada)
        sns.boxplot(
            data=self.df, x='grupo_renda', y='NOTA_MEDIA',
            order=ORDEM_GRUPO_RENDA, palette='coolwarm',
            hue='grupo_renda', dodge=False, legend=False
        )
        plt.title('Distribuição da Nota Média por Grupo de Renda')
        self._salvar_figura('1_boxplot_nota_por_grupo_renda.png')

    def barplot_provas(self) -> None:
        """Gráfico de barras: nota por prova e grupo de renda."""
        df_longo = self.df.melt(
            id_vars=['grupo_renda'], value_vars=COLUNAS_NOTAS,
            var_name='PROVA', value_name='NOTA'
        )
        df_longo['PROVA'] = df_longo['PROVA'].map(PROVAS_NOMES)

        plt.figure(figsize=self.config.figsize_longa)
        sns.barplot(
            data=df_longo, x='PROVA', y='NOTA',
            hue='grupo_renda', hue_order=ORDEM_GRUPO_RENDA,
            palette='coolwarm', errorbar=('ci', 95)
        )
        plt.title('Nota Média por Prova e Grupo de Renda')
        plt.legend(title='Grupo de Renda', bbox_to_anchor=(1.02, 1), loc='upper left')
        self._salvar_figura('2_barplot_nota_por_prova_renda.png')

    def scatter_regressao(self) -> None:
        """Scatterplot com reta de regressão."""
        x = self.df['RENDA_NUMERICA']
        y = self.df['NOTA_MEDIA']
        modelo = sm.OLS(y, sm.add_constant(x)).fit()

        plt.figure(figsize=self.config.figsize_quadrada)
        sns.scatterplot(
            data=self.df.sample(n=min(2000, len(self.df)), random_state=42),
            x='RENDA_NUMERICA', y='NOTA_MEDIA',
            color='gray', alpha=0.12, label='Amostra'
        )
        sns.regplot(data=self.df, x='RENDA_NUMERICA', y='NOTA_MEDIA',
                    scatter=False, color='red', line_kws={'linewidth': 2, 'label': 'Regressão'})
        plt.title('Relação entre Renda Estimada e Nota Média')
        plt.text(x.max() * 0.55, y.min() + 80,
                 f'R² = {modelo.rsquared:.4f}\np = {modelo.pvalues.iloc[1]:.3e}',
                 bbox=dict(facecolor='white', alpha=0.8))
        self._salvar_figura('3_scatterplot_regressao.png')

    def heatmap_correlacao(self) -> None:
        """Heatmap de correlação de Spearman."""
        correlacao = self.df[[
            'RENDA_NUMERICA', 'NU_NOTA_CN', 'NU_NOTA_CH',
            'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NOTA_MEDIA'
        ]].corr(method='spearman')
        labels = ['Renda', 'CN', 'CH', 'LC', 'MT', 'Redação', 'Média']
        correlacao.columns = labels
        correlacao.index = labels

        plt.figure(figsize=(9, 7))
        sns.heatmap(correlacao, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt='.3f')
        plt.title('Correlação de Spearman entre Renda e Notas')
        self._salvar_figura('4_heatmap_correlacao.png')

    def violin_redacao(self) -> None:
        """Violin plot das notas de redação por grupo."""
        plt.figure(figsize=self.config.figsize_quadrada)
        sns.violinplot(
            data=self.df, x='grupo_renda', y='NU_NOTA_REDACAO',
            order=ORDEM_GRUPO_RENDA, palette='coolwarm',
            hue='grupo_renda', dodge=False, legend=False
        )
        plt.title('Distribuição das Notas de Redação por Grupo de Renda')
        self._salvar_figura('5_violinplot_redacao.png')

    def boxplot_internet(self) -> None:
        """Boxplot de matemática por acesso à internet."""
        plt.figure(figsize=self.config.figsize_quadrada)
        sns.boxplot(
            data=self.df, x='INTERNET_BANDA_LARGA_DESCRICAO', y='NU_NOTA_MT',
            order=['Sem banda larga', 'Com banda larga'],
            hue='INTERNET_BANDA_LARGA_DESCRICAO', palette='viridis', legend=False
        )
        plt.title('Notas de Matemática por Acesso à Internet Banda Larga')
        self._salvar_figura('6_boxplot_internet_matematica.png')

    def gerar_todos_graficos(self) -> None:
        """Gera todas as visualizações."""
        logger.info('Gerando visualizações...')
        self.boxplot_renda()
        self.barplot_provas()
        self.scatter_regressao()
        self.heatmap_correlacao()
        self.violin_redacao()
        self.boxplot_internet()
        logger.info('Todas as visualizações foram geradas.')

# =============================================================================
# CLASSE 7: GERADOR DE RELATÓRIO
# =============================================================================

class GeradorRelatorio:
    """
    Classe responsável por gerar o relatório final em Markdown.
    """

    def __init__(self, df: pd.DataFrame, config: ConfigAnalise) -> None:
        self.df = df
        self.config = config

    def gerar(self) -> None:
        """Gera o relatório final com os principais insights."""
        logger.info('Gerando relatório final...')

        nota_media_geral = self.df['NOTA_MEDIA'].mean()
        nota_por_grupo = self.df.groupby('grupo_renda')['NOTA_MEDIA'].mean()
        nota_por_regiao = self.df.groupby('REGIAO')['NOTA_MEDIA'].mean().sort_values(ascending=False)

        internet_mt = self.df.groupby('INTERNET_BANDA_LARGA_DESCRICAO')['NU_NOTA_MT'].mean()

        texto = f"""# Relatório de Análise: Renda Familiar e Desempenho no ENEM 2023

## 📊 Sumário Executivo

Este relatório foi gerado automaticamente pelo pipeline de análise orientado a objetos,
demonstrando a aplicação de POO em um projeto real de ciência de dados.

**Dataset:** {len(self.df):,} candidatos | **Nota média geral:** {nota_media_geral:.1f}

---

## 🔍 Principais Descobertas

### 1. Disparidade por Renda
| Grupo | Nota Média |
|-------|-----------|
"""
        for grupo in ORDEM_GRUPO_RENDA:
            if grupo in nota_por_grupo.index:
                texto += f"| {grupo} | {nota_por_grupo[grupo]:.1f} |\n"

        texto += f"""
### 2. Desigualdade Regional
| Região | Nota Média |
|--------|-----------|
"""
        for regiao in nota_por_regiao.index:
            texto += f"| {regiao} | {nota_por_regiao[regiao]:.1f} |\n"

        texto += f"""
### 3. Impacto da Internet na Matemática
| Acesso | Nota Média |
|--------|-----------|
| Sem banda larga | {internet_mt.get('Sem banda larga', 0):.1f} |
| Com banda larga | {internet_mt.get('Com banda larga', 0):.1f} |

---

*Relatório gerado em {pd.Timestamp.now().strftime('%d/%m/%Y às %H:%M')}*
*Pipeline orientado a objetos desenvolvido como projeto acadêmico de ADS*
"""
        destino = Path('relatorio_final.md')
        with open(destino, 'w', encoding='utf-8') as arquivo:
            arquivo.write(texto)
        logger.info(f'Relatório salvo: {destino}')

# =============================================================================
# PONTO DE ENTRADA PRINCIPAL
# Demonstra: Instanciação e execução do pipeline completo
# =============================================================================

def main() -> None:
    """
    Função principal que orquestra todo o pipeline.
    Demonstra o uso das classes em conjunto.
    """
    # Instancia a configuração
    config = ConfigAnalise()

    # Instancia o pipeline (que herda de AnaliseENEM)
    pipeline = PipelineAnaliseENEM(config)

    # Executa a análise completa
    pipeline.executar_analise()

if __name__ == '__main__':
    main()