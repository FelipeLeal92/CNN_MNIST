# Projeto: Classificação de Dígitos com CNN em TensorFlow

## Visão Geral
Este repositório contém um exemplo completo de projeto em Python para treinar uma **Rede Neural Convolucional (CNN)** visando classificar dígitos manuscritos do dataset **MNIST** usando **TensorFlow/Keras**. É ideal para demonstrar conhecimentos em CNN, Python e TensorFlow em processos seletivos.

---

## Estrutura de Diretórios
cnn-mnist/<br>
├── README.md   # Documentação do projeto <br>
├── requirements.txt   # Dependências <br>
├── train.py   # Script principal de treinamento <br>
└── saved_model/   # Modelo treinado (gerado após execução) 

## 1. Arquivos Principais
### 1.1. `requirements.txt`
### 1.2. `train.py`

## 2. Instruções de Uso
### 2.1. Clone o repositório
- git clone https://github.com/SEU_USUARIO/cnn-mnist.git <br>
- cd cnn-mnist
### 2.2. Crie e ative um ambiente virtual
- python -m venv venv
- source venv/bin/activate  # Linux/Mac
- venv\Scripts\activate     # Windows

### 2.3. Instale as dependências
- pip install -r requirements.txt

### 2.4. Execute o treinamento
- python train.py

### 2.5. Confira o resultado
- Arquivo do modelo salvo em saved_model/cnn_mnist.keras
- Métricas de acurácia impressas no console

## 3. Deploy e Próximos Passos
- Deploy:
  - Use a pasta saved_model/ para servir o modelo via TensorFlow Serving.
  - Ou converta para TensorFlow.js / TensorFlow Lite para aplicações Web ou móveis.

- Melhorias:
  - Ajuste de hiperparâmetros (número de filtros, learning rate).
  - Aumento de dados (data augmentation) com ImageDataGenerator.
  - Testar outras arquiteturas (ResNet, MobileNet).
