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
