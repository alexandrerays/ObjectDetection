# Detecção de Objetos (pt-br)

### Pré-requisitos para execução do projeto:

1. **Conta no gmail:** para poder criar um notebook do Google Colab e treinar o modelo.
2. **Pyenv:** para poder instalar várias versões do Python (Opcional)
3. **Pip:** gerenciador de pacotes para instalar libs do Python
4. **Pipenv:** Gerenciador de variáveis de ambiente para poder criar vários ambientes diferentes.
5. **IDE:** PyCharm, VSCode ou qualquer outro que te deixe mais confortável. Para poder criar e editar códigos. Neste tutorial irei utilizar o PyCharm.

### Preparando o ambiente de desenvolvimento:

Recomendo utilizar Linux/Mac pois esse tutorial não foi testado para o Windows.

1. Criar conta no gmail.

2. As instruções para instalar o pyenv podem ser encontradas [aqui](https://github.com/pyenv/pyenv).

3. Para instalar o pip, seguir as intruções para [Mac](https://www.geeksforgeeks.org/how-to-install-pip-in-macos/), [Linux](https://www.geeksforgeeks.org/how-to-install-pip-in-linux/) e [Windows](https://www.geeksforgeeks.org/how-to-install-pip-on-windows/#:~:text=Step%201%3A%20Download%20the%20get,where%20the%20above%20file%20exists.&text=Step%204%3A%20Now%20wait%20through,Voila!).

4. Para instalar o pipenv, seguir as instruções para [Mac/Linux](https://pipenv.pypa.io/en/latest/) ou [Windows](https://www.pythontutorial.net/python-basics/install-pipenv-windows/).

5. Você pode fazer o download da versão gratuita do PyCharm [aqui](https://www.jetbrains.com/pycharm/download/?source=google&medium=cpc&campaign=14127625370&term=pycharm&gclid=CjwKCAjwtKmaBhBMEiwAyINuwAPqj3d4hEYdFrnIrjFCfaF8ObyNt2guUWoTqWWeRQP_iVBDEA6WoxoCDS8QAvD_BwE#section=mac). Já o download do VSCode pode ser encontrado [aqui](https://code.visualstudio.com/download).

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


