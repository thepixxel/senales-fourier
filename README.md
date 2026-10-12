# Simulación y análisis de señales con la transformada de Fourier

Actividad formativa 2 de la materia **Señales y Sistemas**
Ingeniería en Desarrollo de Software - Universidad Ciudadana de Nuevo León
Autor: Israel Florido Montero

## Propósito

Este programa en Python sirve para ver cómo se comportan algunas señales en el
dominio del tiempo y en el dominio de la frecuencia usando la transformada de
Fourier (calculada con `np.fft.fft()`).

El código hace lo siguiente:

1. Crea tres señales elementales: pulso rectangular, escalón (amortiguado) y senoidal.
2. Calcula su transformada de Fourier y grafica la magnitud y la fase del espectro.
3. Compara los resultados numéricos contra las fórmulas teóricas del libro de Kamen y Heck (2008).
4. Verifica tres propiedades de la transformada: **linealidad**, **desplazamiento en el tiempo** y **escalamiento**.
5. Aplica la propiedad de modulación a un ejemplo real: **modulación de amplitud (AM)**.

## Requisitos

- Python 3
- NumPy
- Matplotlib

```bash
pip install numpy matplotlib
```

## Cómo ejecutarlo

```bash
python senales_fourier.py
```

Para que solo guarde las imágenes sin abrir ventanas:

```bash
python senales_fourier.py --no-mostrar
```

Las gráficas se guardan en la carpeta `figuras/` y los resultados numéricos
(errores contra la teoría, picos del espectro, etc.) se imprimen en la terminal.

## Estructura del código

| Sección | Contenido |
|---|---|
| 0 | Configuración general y colores de las gráficas |
| 1 | Funciones auxiliares (señales, transformada con FFT, gráficas) |
| 2 | Definición de las señales en el dominio del tiempo |
| 3 | Transformada de Fourier de cada señal |
| 4 | Verificación de propiedades |
| 5 | Aplicación real: modulación AM |
