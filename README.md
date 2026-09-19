# Atividades de Processamento de Dados — UFRPE

Resoluções das listas extraclasse da disciplina, implementadas em Python com
NumPy. Todas as questões usam **operações vetorizadas**, sem `for` ou `while`
para percorrer os dados.

## Estrutura

| Pasta | Conteúdo |
|-------|----------|
| `AT1/` | Lista 1 — operações vetorizadas sobre vetores e matrizes |
| `AT2/` | Lista 2 — processamento de imagens monocromáticas |

Cada questão fica em um arquivo próprio (`Q1.py`, `Q2.py`, …), executável de
forma independente.

Na `AT2/`, as imagens de entrada ficam em `imagens de entrada/` e cada script
grava seus resultados numa pasta derivada do próprio nome (`Q1.py` → `Q1_output/`).

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

Os arquivos de questão são **autocontidos**: cada um traz as funções de leitura e
gravação que utiliza, e pode ser executado sozinho, sem depender dos outros.

## Organização em branches

Cada lista é entregue em sua própria branch:

| Branch | Lista |
|--------|-------|
| `LEC1_EugenioAraujo` | AT1 |
| `LEC2_EugenioAraujo` | AT2 |

## Como executar

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt

python AT2/Q1.py
```

## Dependências

`numpy`, `scipy`, `matplotlib` e `opencv-python` — versões fixadas em
[requirements.txt](requirements.txt).

---

# Enunciados — Lista Extra Classe 2

## Q1. Transformação de Intensidade

Dada uma **(a)** imagem monocromática, transformar o espaço de intensidades
(níveis de cinza) para:

- **b)** obter o negativo da imagem, ou seja, o nível de cinza 0 será convertido
  para 255, o nível 1 para 254 e assim por diante;
- **c)** espelhar verticalmente a imagem original;
- **d)** converter o intervalo de intensidades para [100, 200];
- **e)** inverter os valores dos pixels das linhas pares da imagem, ou seja, os
  valores dos pixels da linha 0 serão posicionados da direita para a esquerda, os
  valores dos pixels da linha 2 serão posicionados da direita para a esquerda e
  assim por diante;
- **f)** espelhar as linhas da metade superior da imagem na parte inferior da
  imagem.

## Q2. Ajuste de Brilho

Aplicar a correção gama para ajustar o brilho de uma imagem monocromática `A` de
entrada e gerar uma imagem monocromática `B` de saída. A transformação pode ser
realizada:

- **i)** convertendo-se as intensidades dos pixels do intervalo [0, 255] para
  [0, 1];
- **ii)** aplicando-se a equação `B = A^(1/γ)`;
- **iii)** convertendo-se os valores resultantes de volta para o intervalo
  [0, 255].

Realizar a correção com diferentes valores de `γ` (na figura do enunciado:
1.5, 2.5 e 3.5).

## Q3. Binarização por Limiar (Limiarização)

Dada uma imagem monocromática `A`, gerar uma imagem binária `B` a partir de um
limiar fixo `T`. A operação de conversão é dada por:

```
B = 255, se A > T
B = 0,   caso contrário
```

Na figura do enunciado, `T = 128`.

## Q4. Planos de Bits

Extrair os planos de bits de uma imagem monocromática. Os níveis de cinza de uma
imagem monocromática com `m` bits podem ser representados na forma de um
polinômio de base 2:

```
a_(m-1)·2^(m-1) + a_(m-2)·2^(m-2) + ... + a_1·2^1 + a_0·2^0
```

O plano de bits de ordem 0 é formado pelos coeficientes `a_0` de cada pixel,
enquanto o plano de bits de ordem `m − 1` é formado pelos coeficientes
`a_(m-1)`.

## Q5. Mosaico

Construir um mosaico de 4 × 4 blocos a partir de uma imagem monocromática. A
disposição dos blocos deve seguir a numeração mostrada na figura (c) do
enunciado:

| | | | |
|---|---|---|---|
| 6 | 11 | 13 | 3 |
| 8 | 16 | 1 | 9 |
| 12 | 14 | 2 | 7 |
| 4 | 15 | 10 | 5 |

Os números correspondem à posição original de cada bloco, contada da esquerda
para a direita e de cima para baixo.

## Q6. Combinação de Imagens

Combinar duas imagens monocromáticas de mesmo tamanho por meio da média
ponderada de seus níveis de cinza. Na figura do enunciado, as combinações
apresentadas são `0.2·A + 0.8·B`, `0.5·A + 0.5·B` e `0.8·A + 0.2·B`.
