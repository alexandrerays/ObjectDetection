# Detecção de Objetos (pt-br)

### Pré-requisitos para execução do projeto:

As instruções para configuração do ambiente do tutorial podem ser encontradas [aqui](https://gist.github.com/alexandrerays/63f8d23115d9daa0096075fbf84f56d2).

### Como rodar (para Linux/Mac):

1. Preparar o ambiente de desenvolvimento conforme os requisitos acima.

2. Clonar o repositório do projeto. Para fazer isso abra o terminal e digite:

```
git clone https://github.com/alexandrerays/ObjectDetection
```

4. Baixar os pesos da rede neural pré-treinada e o arquivo de configuração da arquitetura da rede [aqui](https://drive.google.com/drive/folders/1TIC9cVfInS-78bDaWeCmaNQJtYgclYfu?usp=sharing) e salvar na pasta `models/`.

5. Criar uma env (virtual environment) usando o Pipenv especificando a versão do Python:

```shell
pipenv --python 3.8
```

5. Instalar as dependências do projeto

```shell
pipenv install
```

6. Rodar o streamlit.

```shell
streamlit run ./src/main.py
```

# O que é o que neste projeto?

* `images/`: Imagens utilizadas para este tutorial.
* `input/`: Onde estão localizadas as imagens e vídeos para testarmos. Podemos baixar da internet e usar outros artefatos se quiser e colocar nesta pasta.
* `labels/`: Todas as possíveis categorias que o modelo reconhece.
* `model/`: Onde está o arquivo de configuração da arquitetura da rede neural e os pesos pré-treinados.
* `src/`: Códigos para execução do projeto.
* `Pipfile` e `Pipfile.lock`: Arquivos com as dependências do projeto (libs utilizadas).

# Referências

* [Forked Repository](https://github.com/zhoroh/ObjectDetection)


