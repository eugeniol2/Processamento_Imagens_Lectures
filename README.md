# Atividades de Processamento de Dados — UFRPE

Resoluções das listas extraclasse da disciplina, implementadas em Python com
NumPy. As questões usam **operações vetorizadas**, sem `for` ou `while` para
percorrer os dados, exceto quando o próprio enunciado descreve o algoritmo com
laços (é o caso da rotação na Q1 da AT3).

## Estrutura

Esta branch contém a Lista 3:

| Caminho | Conteúdo |
|---------|----------|
| `AT3/Q1.py` … `AT3/Q5.py` | uma questão por arquivo, executável de forma independente |
| `AT3/utils.py` | funções de leitura e gravação de imagens, usadas por todas as questões |
| `AT3/imagens de entrada/` | imagens fornecidas com o enunciado |
| `AT3/Qn_output/` | resultados de cada questão, em pasta derivada do nome do script |
| `AT3/Q3.docx` | análise dos efeitos dos filtros da Q3 |

## Organização do código

Cada operação foi escrita como uma **função nomeada**, em vez de código solto no
corpo do script. A intenção é que o arquivo continue legível meses depois: o nome
da função diz o que ela faz, e o bloco final — depois da linha de `====` — mostra
a sequência de chamadas, funcionando como um resumo da questão.

As funções seguem três regras:

- **Recebem e devolvem arrays NumPy.** Nenhuma delas lê ou grava arquivo; a
  entrada e a saída em disco ficam isoladas em `readImageFromMemory` e
  `saveImage`. Assim a mesma transformação serve para qualquer imagem.
- **Não modificam o argumento recebido.** As que precisam escrever em posições
  específicas trabalham sobre uma cópia, de modo que a imagem original continue
  disponível para as demais operações do mesmo script.
- **São genéricas quanto ao tamanho.** Dimensões e limites vêm sempre de
  `shape`, nunca de valores fixos no código.

Cada passo do enunciado tem **sua própria função**, e o bloco final as encadeia
uma linha por passo. Na Q4, por exemplo:

```python
watchPadded = paddingByRepeatingBorder(watchImg, kernelSize=21)
watchBlurred = applyGaussianBlur(watchPadded, kernelSize=21, sigma=4)
watchRatio = divideByBlurredVersion(watchImg, watchBlurred)
watchSketch = convertRatioToGrayLevels(watchRatio)
```

As contas são feitas em ponto flutuante, porque os valores intermediários podem
ser negativos, passar de 255 ou ter casas decimais. Só no último passo o
resultado é levado para a faixa 0–255, arredondado e convertido para `uint8`,
que é o formato de uma imagem em tons de cinza.

Os arquivos de questão são **autocontidos**: toda a lógica de uma questão fica no
próprio arquivo, e ele pode ser executado sozinho, sem depender das outras
questões.

A única exceção são as duas funções de entrada e saída, `readImageFromMemory` e
`saveImage`. Como são idênticas em todas as questões e não fazem parte do que
cada questão resolve, a partir da `AT3/` elas ficam em `utils.py`, na mesma pasta
das questões, e são importadas por elas.

## Organização em branches

Cada lista é entregue em sua própria branch:

| Branch | Lista |
|--------|-------|
| `LEC1_EugenioAraujo` | AT1 |
| `LEC2_EugenioAraujo` | AT2 |
| `LEC3_EugenioAraujo` | AT3 |

## Como executar

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt

python AT3/Q1.py
```

## Dependências

`numpy`, `scipy`, `matplotlib` e `opencv-python` — versões fixadas em
[requirements.txt](requirements.txt).

---

# Enunciados — Lista Extra Classe 3

## Q1. Rotação de Imagem em múltiplos de 90°

Dada uma imagem monocromática, implementar a rotação da imagem nos seguintes
ângulos:

- **a)** 90° no sentido horário;
- **b)** 180°;
- **c)** 270° no sentido horário.

A rotação deve ser realizada sem utilizar funções prontas de rotação das
bibliotecas, ou seja, a implementação deve manipular diretamente os índices da
matriz da imagem. A imagem utilizada é a `baboon_monocromatica.png`.

O enunciado descreve o algoritmo: inicializar uma matriz vazia, percorrer todos
os elementos da original com dois laços aninhados e colocar cada elemento
`(i, j)` na nova posição, onde `n` é o tamanho da matriz:

| Rotação | Nova posição de `(i, j)` |
|---------|--------------------------|
| 90° horário | `(j, n − i − 1)` |
| 180° | `(n − i − 1, n − j − 1)` |
| 270° horário | `(n − j − 1, i)` |

## Q2. Ampliação da Imagem usando a técnica de vizinho mais próximo

A partir de uma imagem monocromática, gerar uma imagem ampliada utilizando a
técnica de vizinho mais próximo:

- **a)** ampliação por um fator de 2;
- **b)** ampliação por um fator de 4.

A imagem utilizada é a `baboon_monocromatica.png`.

Cada pixel da imagem ampliada recebe o valor do pixel mais próximo da imagem
original. A posição `(i, j)` da imagem ampliada é projetada de volta para a
original e arredondada para o inteiro mais próximo, onde `n` é o tamanho da
original e `m` o da ampliada:

```
x = i · n / m        y = j · n / m
x' = round(x)        y' = round(y)
saída(i, j) = entrada(x', y')
```

## Q3. Filtragem de Imagens

A filtragem aplicada a uma imagem digital é uma operação local que altera os
valores de intensidade dos pixels levando em conta tanto o valor do pixel em
questão quanto os valores dos pixels vizinhos. Utiliza-se uma operação de
convolução (mais precisamente, correlação) de uma máscara pela imagem, o que
equivale a percorrer toda a imagem alterando seus valores conforme os pesos da
máscara e as intensidades da imagem.

Aplicar à imagem `aerial_view.png`:

- **Filtros de caixa** com máscaras 3×3, 5×5 e 7×7, com todos os pesos iguais a
  1 e normalizadas por 1/9, 1/25 e 1/49.
- **Filtros gaussianos** com as máscaras abaixo (referência complementar:
  <https://patrickemmettfuller.com/gaussian-blur/>):

`h4` = 1/16 ×

```
1  2  1
2  4  2
1  2  1
```

`h5` = 1/273 ×

```
1   4   7   4   1
4  16  26  16   4
7  26  41  26   7
4  16  26  16   4
1   4   7   4   1
```

`h6` = 1/1003 ×

```
0   0   1    2   1   0  0
0   3  13   22  13   3  0
1  13  59   97  59  13  1
2  22  97  159  97  22  2
1  13  59   97  59  13  1
0   3  13   22  13   3  0
0   0   1    2   1   0  0
```

## Q4. Esboço a Lápis

Implementar um efeito de esboço a lápis em uma imagem por meio dos seguintes
passos:

- **i)** aplicar um filtro de desfoque gaussiano (por exemplo, com uma máscara
  de 21 × 21 pixels) para suavizar os detalhes da imagem;
- **ii)** dividir a imagem em tons de cinza pela versão desfocada para realçar
  os contornos.

A imagem utilizada é a `gray_watch.png`.

## Q5. Bordas com filtro passa-alta de segunda ordem (Laplaciano)

Um filtro laplaciano é um detector de bordas utilizado para calcular as segundas
derivadas de uma imagem, medindo a taxa com que as primeiras derivadas variam.
Isso permite determinar se a variação entre valores de pixels adjacentes
corresponde a uma borda ou a uma transição contínua. Os kernels laplacianos
geralmente contêm valores negativos dispostos em um padrão em cruz, centralizado
na matriz. Os cantos podem ser iguais a zero ou assumir valores positivos, e o
valor central pode ser negativo ou positivo.

Aplicar o kernel 3×3 abaixo à imagem `gray_peppers.png`:

```
0   1   0
1  -4   1
0   1   0
```
