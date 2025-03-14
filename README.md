
# Parcial2 - He2AI

## Autores
[**Luis Alejandro Rubiano**]()

[**Nicolas Martinez Velez**]()

[**Carlos Andres Castillo**]()

[**Catalina Campos**]()

[**Manuela Pineda**]()

## 📚 Descripción  
Este proyecto aplica **redes neuronales** para predecir la categoría del PIB de los países, utilizando datos históricos del World Bank Group. Se transforma un problema de regresión, basado en valores numéricos del PIB, en un problema de **clasificación múltiple** dividiendo los países en tres grupos: **Low**, **Medium** y **High GDP**. La metodología abarca desde el preprocesamiento y la transformación de datos hasta la implementación y evaluación de diversos modelos de redes neuronales.

## 🎯 Planteamiento del Problema
El objetivo de este proyecto es clasificar a los países, en función de su PIB histórico (1960-2022), en tres categorías: **Low GDP**, **Medium GDP** y **High GDP**. Este es un problema **supervisado** de **clasificación múltiple**, en el que la variable objetivo **(GDP_category)** toma uno de tres valores. 

Se busca determinar cuál de los modelos de redes neuronales probados –RN Tradicional, RN Profunda– permite una mejor clasificación, y se analiza también el impacto de configuraciones subóptimas en el desempeño -RN con mala función de pérdida-.

## 🤖 Algoritmos Implementados
A continuación, se describen los modelos de redes neuronales implementados en este proyecto:

1. **Red Neuronal Tradicional con Scikit-Learn**:  
   Es una implementación de un Perceptrón Multicapa (MLP) utilizando Scikit-Learn (MLPClassifier o MLPRegressor). La red cuenta con una capa de entrada, una o más capas ocultas y una capa de salida, y ajusta sus pesos mediante algoritmos de optimización como el descenso de gradiente estocástico (SGD) para minimizar la función de pérdida. Este enfoque es fácil de usar y eficiente, aunque tiene limitaciones para construir redes profundas y ofrece menos opciones de personalización.

2. **Red Neuronal Profunda con TensorFlow**:  
   Utiliza TensorFlow para construir arquitecturas más complejas y profundas. La red se define como una secuencia de capas (Layers) interconectadas, donde los datos fluyen a través de un grafo computacional. Durante el entrenamiento se aplica la retropropagación para actualizar pesos utilizando optimizadores como Adam, permitiendo aprovechar hardware especializado (por ejemplo, GPUs) y obtener mayor flexibilidad en el diseño del modelo.

3. **Red Neuronal con Función de Pérdida Inadecuada**:  
   Este modelo se configura intencionalmente con una función de pérdida inapropiada (por ejemplo, MSE en un problema de clasificación) y un learning rate demasiado alto. Este enfoque permite observar cómo una elección inadecuada de la función de pérdida y de los hiperparámetros afecta negativamente la convergencia y el rendimiento del modelo, sirviendo como estudio de caso para la importancia de configurar correctamente estos parámetros.

   ## 📂 Contenido del Repositorio

- **GDP_script.ipynb**: El script principal para ejecutar todos los modelos y evaluar su rendimiento.
- **README.md**: Este archivo que estás leyendo.
- 
   
# 📖 Relatoria 

Cronología del desarrollo

**Preguntas iniciales y problemas identificados**

* Transformar el PIB en una variable categórica: El parcial requería convertir un valor continuo (PIB) en tres categorías (por ejemplo, alto, medio y bajo). Este cambio de paradigma implicaba revisar la forma de evaluar la precisión y requería planificar el preprocesamiento (normalización, logaritmos, etc.) para asegurar resultados consistentes.}

* Selección de métricas para la nueva variable objetivo: El equipo debatió sobre cuál métrica sería la más apropiada tras el cambio a clasificación: exactitud (accuracy), F1-score, matriz de confusión y curva ROC, entre otras. Finalmente, acordamos enfocarnos en la precisión (accuracy) y complementar con la matriz de confusión y el F1-score, para entender mejor el desempeño del modelo.

* Normalización y preprocesamiento de datos: Reconocimos que no era trivial decidir si normalizar las variables numéricas antes o después de la conversión a clasificación. Tras debatir, concluimos que lo óptimo era realizar una normalización (por ejemplo, escala estandar o logarítmica) antes de entrenar los modelos, conservando la integridad del dataset.

* Manejo de valores faltantes y coherencia temporal: Encontramos muchos datos faltantes (missing values) en años específicos o en países con registros parciales. Además, no tenía sentido utilizar información de años posteriores para predecir años anteriores, así que tuvimos que reordenar cronológicamente el dataset y descartar (o interpolar) datos excesivamente incompletos.

* Integración de un nuevo dataset de expectativa de vida: Decidimos agregar la variable de expectativa de vida para cumplir con el bono propuesto y darle mayor riqueza al análisis. esto exigió más coherencia temporal y más ajustes en la limpieza de datos.

**Soluciones y conclusiones a las que llegamos**

* Interpolación y promedio para missing values: Luego de discutir eliminar datos incompletos, concluimos que era mejor conservar la mayor cantidad de información posible. Adoptamos la interpolación lineal seguida de un promedio para rellenar vacíos. Esta estrategia nos permitió reducir la pérdida de datos sin afectar drásticamente su distribución.

* Limitación temporal (año 1990 en adelante): Para garantizar coherencia entre la información de PIB y la de expectativa de vida, decidimos tomar datos desde 1990, descartando años con demasiados valores faltantes.

* OneHotEncoding para variables categóricas: Implementamos la codificación de variables como regiones, grupos de ingreso, continentes y hemisferios. Esto evitó supuestos de ordinalidad inexistentes y facilitó que los modelos de clasificación procesaran efectivamente estas variables.

* Transformación logarítmica en el PIB: Para estabilizar la escala y mejorar la separación entre categorías, aplicamos logaritmos al PIB antes de clasificar en bajo, medio y alto.

* Selección de métricas de evaluación: Consideramos la precisión (accuracy) como la métrica principal de desempeño, complementándola con reportes de clasificación y curvas ROC.

Conclusiones clave: El manejo de los datos resultó fundamental para el desarrollo. Aprendimos que la estructura temporal es crítica y que la imputación adecuada de datos faltantes, acompañada de normalización y codificación correctas, facilita modelos más robustos.

**Proceso de Construcción y Optimización de los Modelos**

+ **Red neuronal tradicional (Scikit-Learn - Catalina)**: Inicialmente mostró una precisión baja (56%). se procedio a una búsqueda detallada de hiperparámetros usando GridSearchCV, destacando especialmente la importancia de las iteraciones y la cantidad de capas ocultas. Tras estos ajustes, alcanzamos finalmente una precisión del 86% en el conjunto de prueba.

+ **Red neuronal profunda (TensorFlow - Nicolás)**: Se desarrolló un modelo profundo utilizando TensorFlow y realizó una optimización exhaustiva con Keras Tuner. Nicolás definió una función para explorar múltiples configuraciones técnicas (capas ocultas, funciones de activación, dropout, regularización L2 y tasa de aprendizaje), realizando 10 iteraciones automáticas. La mejor configuración obtenida incluyó cuatro capas ocultas, función de activación ReLU, dropout moderado (0.2), regularización L2 leve y un learning rate de 0.01. Este proceso riguroso llevó a obtener finalmente una precisión del 74% en el conjunto de prueba.

+ **Modelo mal configurado**: Implementamos intencionalmente un modelo incorrecto utilizando MLPRegressor con una función de pérdida diseñada para regresión (cuadrática), en lugar de una función apropiada para clasificación (como cross-entropy). Además, utilizamos un learning rate excesivamente alto, lo cual causó problemas serios de convergencia. Esta configuración errónea resultó en una precisión extremadamente baja (33%), demostrando claramente la importancia técnica crítica de elegir adecuadamente tanto la función de pérdida como el learning rate según la tarea específica (clasificación en este caso).

## **División del trabajo** 
Todos los integrantes conocen y ejecutaron de manera independiente el codigo del proyecto. Sin embargo, cada uno lideró una tarea distinta

  **Catalina:** Se centró en la red neuronal tradicional (Scikit-Learn). Inició con la primera arquitectura Perceptron y MLP y realizó los experimentos iniciales, encontrando una precisión inicial del 56%. hizo ajustes de hiperparámetros esenciales (número de capas ocultas, neuronas y número de iteraciones), logrando mejoras significativas en la precisión.

  **Nicolás:** Estuvo a cargo de la red neuronal profunda (TensorFlow), empleando Keras Tuner para la búsqueda de hiperparámetros. Probó distintas configuraciones (capas, dropout, funciones de activación, regularización L2, tasas de aprendizaje, etc.) hasta encontrar una combinación óptima que llevó a una precisión final de 74%. Asimismo diseñó el modelo mal configurado para ilustrar la importancia de usar la función de pérdida adecuada. Además, lideró el análisis con SHAP Values, profundizando en la interpretabilidad de los modelos.

  **Luis:** Lideró el preprocesamiento e integración de datos, incluyendo la detección y resolución de valores faltantes, y la reorganización cronológica del dataset. Fue quien propuso y aplicó OneHotEncoding a las variables categóricas, y la transformación logarítmica del PIB para estabilizar su distribución. Tambien ayudó en la búsqueda de hiperparámetros para el modelo tradicional mediante GridSearchCV. 

  **Manuela:** Asumió la documentación del proyecto y la Relatoria de este, trato de mostrar con precisión las discusiones más importantes y la lógica tras cada decisión técnica.

  **Carlos:** se encargo preguntas adicionales del parcial y los bonos, evaluando la coherencia de los resultados. También profundizó en el uso de herramientas de análisis como la matriz de confusión, la curva ROC y los valores de SHAP, confirmando la solidez de las conclusiones. Realizo una relatoría completa del procesamiento de datos.
