# Proyecto clasificación de personajes de One Piece con Deep Learning

## Motivación y elección del dataset

Para este proyecto quise trabajar con un tema que me resultara interesante y que, al mismo tiempo, permitiera aplicar los contenidos de Deep Learning revisados durante el curso. Finalmente decidí desarrollar un clasificador de personajes de One Piece, utilizando un dataset de imágenes disponible en Kaggle.

La elección del tema nace principalmente de mi interés por la serie, pero también porque presenta un problema de clasificación que no es tan directo como podría parecer. Aunque los personajes tienen características propias, todos pertenecen al mismo estilo de animación y comparten una base visual parecida, como la forma de los rostros, los ojos, el cuerpo y el tipo de dibujo.

En muchos casos las diferencias se encuentran en elementos más específicos, como el cabello, la ropa, cicatrices, accesorios o colores característicos. Además, las imágenes pueden mostrar a los personajes desde distintos ángulos, con fondos diferentes, parcialmente visibles o acompañados de efectos visuales.

Con todo esto, busco responder la siguiente pregunta:

**¿Es posible utilizar un modelo preentrenado para clasificar correctamente imágenes de 18 personajes de One Piece?**

---

## Definición del problema

El problema corresponde a una clasificación multiclase de imágenes. Cada imagen pertenece a uno de los 18 personajes considerados dentro del dataset y el modelo debe aprender a identificar cuál de ellos aparece.

Las clases utilizadas fueron:

Ace, Akainu, Brook, Chopper, Crocodile, Franky, Jinbei, Kurohige, Law, Luffy, Mihawk, Nami, Rayleigh, Robin, Sanji, Shanks, Usopp y Zoro.

La entrada del modelo corresponde a una imagen y la salida es una de las 18 clases posibles. Durante el entrenamiento, el modelo aprende patrones visuales que le permiten diferenciar a los personajes, como formas, colores, texturas, partes del rostro, vestimenta y accesorios.

La principal dificultad es que estas características no siempre aparecen de forma clara. Un personaje puede estar girado, ocupar solo una parte de la imagen o aparecer con colores distintos debido a la iluminación y los efectos de la escena. También puede ocurrir que dos personajes compartan elementos visuales parecidos, lo que aumenta la posibilidad de confusión.

---

## Descripción del dataset

El dataset utilizado fue obtenido desde Kaggle y contiene imágenes de personajes de One Piece organizadas en carpetas según la clase correspondiente.

**Fuente:** (https://www.kaggle.com/datasets/ibrahimserouis99/one-piece-image-classifier/data)

Durante la revisión inicial se encontraron imágenes normales junto con archivos invertidos. Estos últimos fueron excluidos para evitar trabajar con versiones repetidas de una misma imagen. También se revisó que todos los archivos pudieran abrirse correctamente y no se encontraron imágenes dañadas.

Después de realizar esta limpieza, la base quedó formada por un total de **7.603 imágenes válidas**.

Las imágenes fueron divididas en tres conjuntos:

| Conjunto | Cantidad de imágenes | Proporción |
|---|---:|---:|
| Entrenamiento | 5.310 | 70 % |
| Validación | 1.133 | 15 % |
| Prueba | 1.160 | 15 % |
| **Total** | **7.603** | **100 %** |

La división se realizó manteniendo imágenes de cada personaje en los tres conjuntos. El conjunto de prueba quedó separado durante todo el entrenamiento y solo se utilizó para la evaluación final.

Debido al tamaño del dataset, las imágenes originales no se incluyen dentro del repositorio.

---

## Plan de acción

El proyecto comenzó con la revisión de la estructura del dataset y la cantidad de imágenes disponibles para cada personaje. También se analizaron sus dimensiones, orientación y posibles problemas que pudieran afectar el entrenamiento.

Luego se eliminaron las imágenes invertidas y se comprobó que todos los archivos fueran válidos. Después de esto, las imágenes se separaron en entrenamiento, validación y prueba.

Para preparar los datos se utilizaron transformaciones diferentes según el conjunto. En entrenamiento se aplicaron recortes aleatorios y volteo horizontal, con el objetivo de entregar una mayor variedad de imágenes al modelo. En validación y prueba se utilizaron transformaciones fijas para que la evaluación fuera consistente.

Posteriormente se adaptó un modelo ResNet18 preentrenado para trabajar con las 18 clases del problema. Finalmente, el modelo fue entrenado y evaluado mediante accuracy, pérdida, precision, recall, F1-score, matriz de confusión y una revisión visual de predicciones incorrectas.

---

## Modelo seleccionado

El modelo utilizado fue ResNet18, una red neuronal convolucional preentrenada con imágenes de ImageNet.

Se eligió este modelo porque las redes convolucionales están diseñadas para trabajar con imágenes y pueden aprender características visuales como bordes, formas, texturas y patrones más complejos. Además, utilizar un modelo preentrenado permite aprovechar conocimientos obtenidos anteriormente, evitando comenzar el aprendizaje completamente desde cero.

Para adaptar ResNet18 al proyecto se mantuvieron congeladas sus capas principales y se reemplazó la última capa por una nueva salida de 18 clases. De esta manera, el modelo conservó las características visuales aprendidas previamente y solo entrenó la parte encargada de diferenciar a los personajes.

También se agregó una capa Dropout con valor 0,5 para reducir el riesgo de overfitting.

Una ventaja de esta estrategia es que permite entrenar el modelo utilizando menos tiempo y recursos que una red construida desde cero. Una de sus limitaciones es que, al mantener congeladas las capas principales, las características aprendidas originalmente por ResNet18 no se adaptan completamente al estilo visual de One Piece.

---

## Metodología aplicada

La metodología desarrollada en el notebook fue la siguiente:

1. Definición del problema y pregunta principal.
2. Carga y revisión de la estructura del dataset.
3. Conteo de imágenes por personaje.
4. Eliminación de imágenes invertidas.
5. Revisión de archivos dañados.
6. Análisis de las dimensiones y orientación de las imágenes.
7. Separación en entrenamiento, validación y prueba.
8. Aplicación de transformaciones y normalización.
9. Creación de los DataLoader.
10. Carga de ResNet18 preentrenada.
11. Congelamiento de las capas principales.
12. Adaptación de la capa final para 18 clases.
13. Incorporación de Dropout.
14. Entrenamiento utilizando CrossEntropyLoss y Adam.
15. Incorporación de transformaciones, Dropout y early stopping para reducir el riesgo de overfitting.
16. Evaluación con el conjunto de prueba.
17. Revisión de métricas por personaje.
18. Construcción de la matriz de confusión.
19. Revisión visual de predicciones incorrectas.
20. Interpretación y cierre de resultados.

La configuración principal del entrenamiento fue:

| Parámetro | Valor |
|---|---:|
| Modelo | ResNet18 preentrenada |
| Optimizador | Adam |
| Learning rate | 0,001 |
| Función de pérdida | CrossEntropyLoss |
| Batch size | 32 |
| Épocas máximas | 10 |
| Dropout | 0,5 |
| Paciencia de early stopping | 3 épocas |

---

## Resultados obtenidos

El modelo completó las 10 épocas definidas. Early stopping no llegó a activarse, ya que no se acumularon tres épocas seguidas sin una mejora en la pérdida de validación.

El mejor estado del modelo se obtuvo en la época 9, donde se alcanzó la menor pérdida de validación.

En el conjunto de prueba se obtuvieron los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| Accuracy | 72,41 % |
| Pérdida | 1,0368 |
| F1-score macro | 0,7202 |
| F1-score ponderado | 0,7212 |

El resultado de prueba fue cercano al observado en validación, por lo que el modelo mantuvo un comportamiento estable al enfrentarse a imágenes que no habían sido utilizadas durante el entrenamiento.

Al revisar las métricas por personaje, Nami obtuvo el F1-score más alto con 0,8613, seguido por Brook con 0,8372. Rayleigh, Law y Akainu también presentaron buenos resultados.

Las mayores dificultades aparecieron en Zoro, Franky y Ace. En el caso de Franky, el modelo presentó una precision alta de 0,9355, pero un recall de 0,4328. Esto quiere decir que cuando el modelo clasificó una imagen como Franky normalmente acertó, aunque dejó varias imágenes reales de este personaje clasificadas dentro de otras clases.

La matriz de confusión permitió observar algunas confusiones más específicas. Por ejemplo, varias imágenes de Franky fueron clasificadas como Zoro, Chopper o Nami. También se encontraron confusiones entre Shanks y Mihawk, además de algunos casos de Usopp clasificado como Chopper.

Al revisar visualmente algunas predicciones incorrectas se observó que varios errores aparecen cuando el personaje está girado, parcialmente visible o acompañado de fondos y efectos poco habituales. Aun así, también se encontraron errores en imágenes claras, lo que muestra que el modelo todavía puede relacionar características que se repiten entre distintos personajes.

---

## Conclusiones

A partir del trabajo realizado, se puede concluir que sí fue posible utilizar un modelo preentrenado para clasificar una parte importante de las imágenes correspondientes a los 18 personajes.

El modelo alcanzó un accuracy de 72,41 % en el conjunto de prueba, junto con valores similares de F1-score macro y ponderado. Esto muestra que la diferencia en la cantidad de imágenes de cada personaje no tuvo un efecto demasiado grande en el resultado general.

De todas maneras, era esperable que aparecieran dificultades al diferenciar algunos personajes. Como se planteó al inicio, todos pertenecen al mismo estilo de animación y comparten una base visual parecida. Muchas veces la diferencia depende de características puntuales, como el cabello, la ropa, una cicatriz o un accesorio, por lo que si estos elementos no aparecen claramente la clasificación se vuelve más difícil.

Esto también se observó en las predicciones incorrectas, donde algunas imágenes mostraban personajes girados, parcialmente visibles o acompañados de fondos y efectos visuales. También existieron errores en imágenes más claras, lo que indica que el modelo todavía no logra separar correctamente todos los rasgos visuales.

En general, ResNet18 logró adaptarse de una buena manera al problema y no se observó una señal clara de overfitting durante el entrenamiento. El modelo logró aprender características útiles a partir de las imágenes, mientras que las transformaciones aplicadas, el uso de Dropout y el modelo preentrenado ayudaron a trabajar con la variedad presente en el dataset.

Con todo esto, la pregunta planteada al inicio puede responderse indicando que un modelo preentrenado sí puede clasificar una parte importante de las imágenes de los 18 personajes, aunque todavía existen confusiones cuando las diferencias visuales son pequeñas o cuando las características principales del personaje no se observan claramente.

---

## Limitaciones y posibles mejoras

Una limitación del proyecto es que, aunque se eliminaron las imágenes identificadas como inverted, no se realizó una búsqueda de duplicados perceptuales o imágenes muy similares. Por esto, podría existir algún grado de similitud entre imágenes de entrenamiento, validación y prueba, aunque no se encontró evidencia directa de que esto estuviera ocurriendo.

Como mejora futura, se podría incorporar una etapa de comparación de similitud antes de dividir el dataset, con el objetivo de identificar imágenes repetidas o versiones casi iguales. Estas imágenes podrían eliminarse o mantenerse dentro de un mismo conjunto, evitando que una versión aparezca en entrenamiento y otra muy parecida en validación o prueba. De esta manera, la evaluación representaría con mayor seguridad la capacidad del modelo para clasificar imágenes realmente nuevas.

Además, más adelante se podría ajustar algunas de las capas finales de ResNet18 o incorporar más imágenes para las clases que presentaron mayores dificultades.

---

## Estructura del repositorio

El repositorio contiene los archivos necesarios para revisar el desarrollo del proyecto y reproducir el entorno utilizado:

```text
proyecto-clasificacion-personajes-one-piece/
├── Proyecto_One_Piece_Clasificacion.ipynb
├── README.md
├── environment.yml
└── .gitignore
```

La carpeta Data debe descargarse desde la fuente indicada y ubicarse dentro de la carpeta principal del proyecto. Debido al tamaño de las imágenes, esta carpeta no se incluye en GitHub.

Durante la ejecución del notebook también se generan las carpetas one_piece_limpio y one_piece_dividido, las cuales almacenan las imágenes después de la limpieza y la separación en entrenamiento, validación y prueba. Estas carpetas tampoco se incluyen en el repositorio.

El dataset original también incluye un archivo llamado classnames.txt con los nombres de los personajes. Este archivo no se utiliza en el proyecto, ya que las clases se obtienen directamente desde los nombres de las carpetas mediante ImageFolder, por esto no se incluye dentro del repositorio.

---

## Ejecución del proyecto

Para ejecutar el proyecto primero se debe descargar el dataset desde la fuente indicada y ubicar la carpeta Data dentro de la carpeta principal.

Luego se puede crear el entorno utilizando el archivo environment.yml:

```bash
conda env create -f environment.yml
conda activate one_piece_ia
```

Finalmente, se debe abrir el archivo Proyecto_One_Piece_Clasificacion.ipynb desde Visual Studio Code y seleccionar el entorno one_piece_ia como kernel.

El notebook está preparado para detectar las imágenes si se encuentran dentro de Data o dentro de una segunda carpeta con el mismo nombre.

---

## Dependencias

El proyecto fue desarrollado en Python utilizando principalmente las siguientes librerías:

- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow

Las dependencias necesarias para ejecutar el proyecto se encuentran reportadas en el archivo environment.yml.
