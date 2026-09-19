# Atividades de Processamento de Dados — UFRPE

Resoluções das listas extraclasse da disciplina, implementadas em Python com
NumPy. Todas as questões usam **operações vetorizadas**, sem `for` ou `while`
para percorrer os dados.

## Estrutura

| Pasta | Conteúdo |
|-------|----------|
| `AT1/` | Lista 1 — operações vetorizadas sobre vetores e matrizes |

Cada questão fica em um arquivo próprio (`Q1.py`, `Q2.py`, …), executável de
forma independente.

## Organização em branches

Cada lista é entregue em sua própria branch:

| Branch | Lista |
|--------|-------|
| `LEC1_EugenioAraujo` | AT1 — operações vetorizadas |
| `LEC2_EugenioAraujo` | AT2 — processamento de imagens |

## Organização do código

Cada operação foi escrita como uma **função nomeada**, em vez de código solto no
corpo do script. A intenção é que o arquivo continue legível meses depois: o nome
da função diz o que ela faz, e o bloco final — depois da linha de `====` — reúne
os dados de entrada, as chamadas e os `print`, funcionando como um resumo da
questão.

As funções seguem três regras:

- **Recebem e devolvem arrays NumPy**, sem imprimir nada por conta própria. A
  exibição dos resultados fica toda no bloco final.
- **Não modificam o argumento recebido**, salvo quando a própria questão pede a
  alteração no local.
- **São genéricas quanto ao tamanho.** Dimensões e limites vêm dos próprios
  arrays, nunca de valores fixos no código.

## Como executar

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
pip install -r requirements.txt

python AT1/Q1.py
```

## Dependências

`numpy`, `scipy` e `matplotlib` — versões fixadas em
[requirements.txt](requirements.txt).

---

# Enunciados — Lista Extra Classe 1

## Q1. Contar o número de transições de Falso para Verdadeiro em uma sequência

```python
x = np.array([False, True, False, False, True])
# Saída: 2
```

## Q2. Selecionar apenas números ímpares e elevá-los ao quadrado

```python
x = np.array([1, 2, 3, 4])
# Saída: [1, 9]
```

## Q3. Somar todos os elementos divisíveis por 5

```python
x = np.array([1, 5, 12, 15, 20, 22])
# Saída: 40
```

## Q4. Somar cada n valores em um vetor

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
n = 4
# Saída: [10, 26, 42]
```

## Q5. Trocar todos os valores negativos em um vetor por zeros

```python
x = np.array([-3, -2, -1, 0, 1, 2, 3])
# Saída: [0, 0, 0, 0, 1, 2, 3]
```

## Q6. Contar número de valores pares em um vetor

```python
x = np.array([1, 2, 3, 4, 5, 6, 7])
# Saída: 3
```

## Q7. Calcular a média móvel com uma janela de tamanho n

```python
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n = 4
# Saída: [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
```

## Q8. Normalizar um vetor para que seus elementos somem 1

```python
x = np.array([1, 2, 3, 4])
# Saída: [0.1, 0.2, 0.3, 0.4]
```

## Q9. Calcular a distância euclidiana entre dois vetores

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# Saída: 5.19615
```

## Q10. Verificar se um vetor está ordenado crescentemente

Retorno: `True` ou `False`.

```python
v = np.array([1, 2, 3, 4, 5])
# Saída: True
```

## Q11. Calcular as diferenças consecutivas de um vetor

```python
v = np.array([10, 7, 4, 3, 2])
# Saída: [-3, -3, -1, -1]
```

## Q12. Somar os elementos positivos de um vetor

```python
v = np.array([1, -2, 3, 4, -5, 6, -7, 8, -9])
# Saída: 22
```

## Q13. Verificar se um vetor é um palíndromo

```python
x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
# Saída: True
```

## Q14. Operações entre duas matrizes

Considere duas matrizes `A` e `B`, de dimensão 10 × 10, representadas como
arrays da biblioteca NumPy. Utilizando operações vetorizadas, implemente uma
função que receba as matrizes `A` e `B` e produza:

- **a)** a soma das duas matrizes;
- **b)** a diferença absoluta entre as duas matrizes;
- **c)** a média das duas matrizes;
- **d)** uma matriz contendo, em cada posição, o maior valor entre os
  elementos correspondentes de `A` e `B`.

Para cada operação, apresente a matriz resultante.

## Q15. Somas e médias por linha e por coluna

Considere uma matriz `A` de dimensão 10 × 10. Implemente uma função que
calcule:

- **a)** a soma dos elementos de cada linha;
- **b)** a soma dos elementos de cada coluna;
- **c)** a média dos elementos de cada linha;
- **d)** a média dos elementos de cada coluna.

Apresente os resultados obtidos e indique as dimensões dos arrays resultantes
em cada caso.

> **Dica:** utilize o conceito de `axis` do NumPy para determinar a dimensão
> sobre a qual cada operação deve ser realizada.

## Q16. Operações em um array tridimensional

Considere um array `A` de dimensão 4 × 5 × 3. Implemente uma função que
realize as seguintes operações:

- **a)** calcule a soma dos elementos ao longo de cada uma das três dimensões;
- **b)** calcule a média dos elementos ao longo de cada uma das três dimensões;
- **c)** determine o maior valor presente em cada uma das três dimensões.

Para cada operação, apresente as dimensões dos arrays resultantes e explique
o efeito do parâmetro `axis`.

> **Desafio adicional:** utilizando apenas operações vetorizadas, identifique
> a posição `(i, j, k)` do maior elemento do array.

## Q17. Normalização de uma matriz para o intervalo [0, 1]

Considere uma matriz `A` de dimensão 10 × 10, contendo valores reais.
Implemente uma função que normalize os valores da matriz para o intervalo
[0, 1], de acordo com a expressão:

```
A_norm = (A − A_min) / (A_max − A_min)
```

em que `A_min` e `A_max` correspondem, respectivamente, ao menor e ao maior
valor presente na matriz.

Apresente a matriz original, os valores mínimo e máximo e a matriz
normalizada.

> **Desafio adicional:** modifique sua implementação para normalizar cada
> linha da matriz de forma independente, utilizando o menor e o maior valor
> de cada linha.
