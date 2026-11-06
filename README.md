# rede-neural-2-camadas
Segundo Trabalho do GitHub Individual da disciplina Fundamentos de IA - Prof. Edjard Mota
AL: Gabriel José Lima dos Santos (CB01)

Rede Neural de 2 Camadas (Implementação dos slides de Fei-Fei Li – Lecture 4) Este código implementa o treinamento completo de uma rede neural artificial de duas camadas usando apenas a biblioteca NumPy. A implementação segue exatamente o exemplo apresentado nos slides da professora Fei-Fei Li (CS231n – Lecture 4).

Arquitetura da Rede:

* Camada de entrada: 1000 características (D_in = 1000)
* Camada oculta: 100 neurônios (H = 100) com função de ativação Sigmoid
* Camada de saída: 10 neurônios (D_out = 10) – linear (sem ativação)
* Tamanho do batch: 64 exemplos (N = 64)

Funcionamento passo a passo:

1. Forward Pass (Propagação para frente)
   
* Calcula a ativação da camada oculta:

h = σ(x · W₁)

onde σ é a função sigmoid: 1 / (1 + exp(-z))

Calcula a predição da saída:

ŷ = h · W₂

2. Cálculo da Perda (Loss)

* Utiliza a Soma dos Erros Quadráticos (MSE):

Loss = Σ (ŷ - y)²

3. Backward Pass (Retropropagação – Cálculo dos Gradientes)

* Calcula o gradiente da perda em relação à saída:

∂L/∂ŷ = 2 · (ŷ - y)

* Calcula o gradiente dos pesos da segunda camada (W₂)
* Propaga o gradiente para a camada oculta
* Calcula o gradiente dos pesos da primeira camada (W₁), utilizando a derivada da função sigmoid:

 σ'(z) = σ(z) · (1 - σ(z))
 
4. Atualização dos Pesos (Gradient Descent)

* Os pesos são atualizados com taxa de aprendizado 1e-4:

W₁ = W₁ - 0.0001 · ∇W₁
W₂ = W₂ - 0.0001 · ∇W₂

Observações:

* O código realiza 2000 iterações de treinamento.
* A perda (loss) diminui ao longo das iterações, demonstrando que a rede está aprendendo.
* Toda a implementação (forward + backward + atualização) é feita manualmente, sem uso de frameworks como PyTorch ou TensorFlow, exatamente como mostrado nos slides.
* Para facilitar a visualização, a perda é impressa a cada 100 iterações.
* Nesta implementação, por simplicidade (como mostrado nos slides), a rede não possui termos de bias
