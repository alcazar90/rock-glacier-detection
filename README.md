## Rock Glacier Detection 📡
Proyecto curso MDS7201-1, en conjunto con el Centro de Modelamiento Matemático (CMM)

<p align="center">
<img src="fig/rock-glacier-portrait.png" width="550"/>
</p>

## Tabla de Contenido
 * [Contexto y Problema](#contexto-y-problema)
 * [`rock-glacier-dataset`](#rock-glacier-dataset)
 * [Modelo `Skynet`](#modelo-skynet)
 * [Recursos](#recursos)


## Contexto y Problema

**TODO**

## Rock Glacier Dataset

Repositorio de nuestro _dataset_ [`rock-glacier-dataset`](https://huggingface.co/datasets/alkzar90/rock-glacier-dataset). Fácil de descargar y compartir con la siguiente línea de código utilizando la librería `datasets` de HuggingFace🤗:

```python
from datasets import load_dataset

data = load_dataset('alkzar90/rock-glacier-dataset')
```

## Modelo Skynet

<a href="https://colab.research.google.com/drive/1QAMQJlxkQilh_Km_oGtgQtZF6FZ6uPvS?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

En primera instancia entrenamos un clasificador que responda la pregunta:
¿hay un glaciar o no en la imagen? El proceso de entrenamiento y _workflow_
se puede encontrar en el Google colab de arriba. Utilizamos _transfer learning_
sobre  un modelo Visual Transformer.

<center>

![](https://huggingface.co/blog/assets/51_fine_tune_vit/vit-figure.jpg)

</center>

El modelo y un API de inferencia para ver como funciona sobre algunos ejemplos
se puede encontrar en el repositorio del modelo [`Skynet`](https://huggingface.co/alkzar90/skynet).

La siguiente versión del modelo es diseñar un modelo de segmentación de 
imagen: ¿este _pixel_ contiene un glaciar?

<br>

## Recursos

**TODO**


```
@misc{MDS8201-1-proyecto,
  authors = {Alcázar, Cristóbal}, {Cortez, Diego}, {Stears, Christopher}, {Subiabre, Cristóbal}
  title = {Rock Glacier Detection using SkyNet},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alcazar90/rock-glacier-detection}},
}
```
