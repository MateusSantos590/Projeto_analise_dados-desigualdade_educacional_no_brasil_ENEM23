# Relatório de Análise: Renda Familiar e Desempenho no ENEM 2023

## 📊 Sumário Executivo

Este relatório analisa a relação entre renda familiar e desempenho no ENEM 2023,
investigando disparidades socioeconômicas no acesso e qualidade educacional.

**Dataset:** 109,178 candidatos | Períodos: jan-nov 2023 | Faixa de renda: A-Q

---

## 🔍 Descobertas Principais

### 1. Disparidade Extrema por Renda
- **Faixa A (Sem renda):** média de 448.6 pontos
- **Faixa Q (Renda + R$26.4k):** média de 755.1 pontos
- **Diferença Q-A:** 306.4 pontos (68.3% acima)

### 2. Impacto da Educação Privada
- **Vantagem escola privada:** +148.4 pontos em média
- **Após controlar fatores:** +73.4 pontos (mantém-se significante)
- **Elite (P90+) em privada:** 75.6%
- **Vulneráveis (P10-) em privada:** 0.1%

### 3. Desigualdade Racial
- **Pretos e pardos no P10-:** 73.8%
- Sobrerrepresentados entre os vulneráveis

### 4. Disparidades Regionais
- **Melhor desempenho:** Norte (513.03)
- **Pior desempenho:** Sul (511.37)
- **Diferença regional:** 1.66 pontos

---

## 📈 Modelo Multivariado (OLS)

### Ajuste do Modelo
- **R²:** 0.7129
- **R² ajustado:** 0.7129
- **F-statistic:** 20849.90 (p < 0.001)
- **Observações:** 109,178.0

### Efeitos Estimados
- **Renda (por R$1.000):** +12.91 pontos
- **Escola privada:** +73.4 pontos
- Ambos significantes ao nível 5%

---

## 🎯 Conclusões

1. **Renda é fator determinante** de desempenho, com efeito robusto e não confundido
2. **Educação privada amplifica vantagem inicial** de renda familiar
3. **Interseccionalidade renda + raça + região** cria clusters de vulnerabilidade
4. **Intervenções devem focar** em: acesso a educação de qualidade, foco em baixa renda + minorias

---

*Análise realizada em 27/05/2026 às 21:32*
*Analista de Dados | Metodologia: EDA + Testes Inferenciais + Regressão Multivariada*
