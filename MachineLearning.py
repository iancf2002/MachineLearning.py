import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

 # Dados de entrada (X): horas de estudo
 # reshape ( -1 ,1) transforma em uma matriz coluna
X = np.array([1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 , 9]).reshape( -1 ,1)

 # Dados de saída (y): notas obtidas
y = np.array([10, 18, 28, 39, 49, 61, 69, 81, 90])

 # Criando o modelo de regressão linear
modelo = LinearRegression()

 # " Treinando " o modelo com os dados
modelo.fit(X , y )

a = modelo.coef_[0]       # inclinação da reta
b = modelo.intercept_     # ponto onde a reta cruza o eixo y

print(f"Equação da reta: y = {a:.2f}x + {b:.2f}")

previsao = modelo.predict(np.array([[23]]))
print(f"Preço previsto para 12 unidades: R$ {previsao[0]:.2f}")

# # Gerando previsões ( reta de regressão )
y_pred = modelo.predict( X )

#  # Plotando os pontos reais ( azul )
plt.scatter(X , y , color ="blue", label ="Dados reais")

#  # Plotando a reta aprendida pelo modelo ( vermelho )
plt.plot (X , y_pred , color ="red", label ="RegressãoLinear ")
# Exibindo gráfico
plt.legend()
plt.show()