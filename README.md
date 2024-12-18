<div align="center">
   <h1><b>Tradutor de Libras</b></h1><br><br>

   <a href="" target="_blank">![License](https://img.shields.io/badge/license-MIT-blue.svg)</a>
   ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)

</div>


## 📖 Descrição

**SignTranslator** é uma aplicação de tradução em tempo real da Língua Brasileira de Sinais (Libras) desenvolvida em Python. Utilizando a webcam, a aplicação analisa os movimentos das mãos do usuário e identifica as letras correspondentes em Libras, facilitando a comunicação entre surdos e ouvintes através de tecnologia avançada de visão computacional e aprendizado de máquina.

<br>

## ⚙️ Funcionalidades

- **Detecção de Mãos:** Identifica e rastreia as mãos do usuário em tempo real usando a webcam.
- **Reconhecimento de Letras em Libras:** Analisa os gestos das mãos e identifica a letra correspondente na Libras.
- **Interface Gráfica Intuitiva:** Interface amigável para iniciar e monitorar o processo de reconhecimento.
- **Treinamento de Modelo Personalizado:** Permite treinar o modelo com um conjunto de dados específico para melhorar a precisão.
- **Previsão em Tempo Real:** Exibe a letra identificada diretamente na interface durante o uso.

<br>

## 🛠 Tecnologias Utilizadas

- **Python 3.8+**
- **OpenCV:** Biblioteca de visão computacional para captura e processamento de vídeo.
- **Mediapipe:** Framework para detecção e rastreamento de mãos.
- **TensorFlow/Keras:** Frameworks de aprendizado de máquina utilizados para treinar e executar o modelo de reconhecimento.
- **NumPy:** Manipulação de arrays e operações matemáticas.
- **Tkinter:** Biblioteca para criação da interface gráfica.
- **Pandas:** Manipulação e análise de dados.

<br>

## 💾 Instalação

### Pré-requisitos

- **Python 3.8 ou superior** instalado em sua máquina. [Baixe aqui](https://www.python.org/downloads/).
- **pip** para gerenciar pacotes Python.

### Passos de Instalação

1. **Clone este repositório:**

   ```bash
   git clone https://github.com/LuccaGiovane/SignTranslator.git
   ```
2. **Navegue até o diretório do projeto:**
   ```bash
   cd SignTranslator
   ```
3. **Crie e ative um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv venv
   ```
   3.1. **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   3.2. **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```
4. **Instale as dependências necessárias:**
   ```bash
   pip install -r requirements.txt
   ```
   
<br>

## 🚀 Uso
### Treinamento do Modelo
Antes de utilizar a aplicação para reconhecimento em tempo real, é necessário treinar o modelo com os dados de sinais em Libras.

1. Prepare o Dataset:
- Baixe o dataset [Libras CNN](https://www.kaggle.com/datasets/allanpardinho/libras-cnn) da Kaggle.
- Extraia e organize os dados nos diretórios dataset/train e dataset/test conforme a estrutura esperada pelo script train.py.

2. **Execute o Script de Treinamento:**
   ```bash
   python train.py
   ```
- O treinamento pode levar algum tempo dependendo do seu hardware.
- Após o treinamento, o modelo será salvo como model.h5.
  
<br>

## 👨🏻‍💻 Executando a Aplicação
1. Inicie a interface Gráfica:
   ```bash
   python main.py
   ```
2. Na Interface:
- Clique no botão "Iniciar" para começar o reconhecimento em tempo real.
- A aplicação acessará a webcam e começará a identificar as letras em Libras baseadas nos gestos das suas mãos.
- Pressione 'q' na janela de vídeo para encerrar o reconhecimento.

<br>

## 📊 Dataset de Treinamento para Libras
Foi utilizado o dataset [Libras CNN](https://www.kaggle.com/datasets/allanpardinho/libras-cnn), que contém milhares de imagens de mãos em diferentes posições correspondentes às letras da Libras. Este dataset foi fundamental para treinar o modelo de reconhecimento, garantindo maior precisão e eficiência na identificação dos gestos.

<br>

## 🤝 Contribuição
Contribuições são bem-vindas! Se você deseja melhorar este projeto, siga as etapas abaixo:

1. Fork este repositório.
1. Crie uma branch para sua feature:
   ```bash
   git checkout -b feature/nova-feature
   ```
3. Faça commit das suas alterações:
   ```bash
   git commit -m "Adiciona nova feature"
   ```
4. Envie para a branch:
   ```bash
   git push origin feature/nova-feature
   ```
5. abra um Pull Request.

<br> 

## 📂 Estrutura do Projeto
```bash
SignTranslator/
│
├── img/
│   └── octocat.png
│
├── scripts/
│   ├── hand_recognition.py
│   ├── train.py
│   └── model.h5
│
├── dataset/
│   ├── train/
│   └── test/
│
├── main.py
├── README.md
└── requirements.txt
```

<br>

- **img/:** Diretório para imagens utilizadas na interface gráfica.
- **scripts/:** Scripts para reconhecimento de mãos e treinamento do modelo.
- **dataset/:** Conjunto de dados para treinamento e teste.
- **main.py:** Script principal para a interface gráfica.
- **requirements.txt:** Lista de dependências do projeto.
