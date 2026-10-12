# -*- coding: utf-8 -*-
"""
Actividad formativa 2: Simulación y análisis de señales con la transformada de Fourier
Materia: Señales y Sistemas
Autor: Israel Florido Montero
Universidad Ciudadana de Nuevo León

Qué hace este programa:
  1. Crea tres señales elementales: pulso rectangular, escalón y senoidal.
  2. Calcula su transformada de Fourier con np.fft.fft().
  3. Grafica la señal en el tiempo y su espectro (magnitud y fase).
  4. Verifica tres propiedades: linealidad, desplazamiento en el tiempo y
     escalamiento.
  5. Aplica la propiedad de modulación a un ejemplo real (radio AM).

Las ecuaciones que se mencionan salen del capítulo 3 del libro de
Kamen y Heck (2008), "Fundamentos de señales y sistemas".

Cómo se ejecuta:
    python senales_fourier.py              (muestra las gráficas)
    python senales_fourier.py --no-mostrar (solo guarda las imágenes)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# 0. CONFIGURACIÓN GENERAL
# ======================================================================

# Las gráficas se guardan en una carpeta para poder usarlas en el informe
CARPETA_FIGURAS = "figuras"
os.makedirs(CARPETA_FIGURAS, exist_ok=True)

# Si pongo --no-mostrar el programa no abre ventanas, solo guarda las imágenes
MOSTRAR = "--no-mostrar" not in sys.argv

# Paleta de colores del informe (verde oscuro y naranja)
VERDE = "#1B4332"
VERDE_CLARO = "#52B788"
NARANJA = "#F28C28"
GRIS = "#6B7280"

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


# ======================================================================
# 1. FUNCIONES AUXILIARES
# ======================================================================

def pulso_rectangular(t, tau=1.0):
    """
    Pulso rectangular p_tau(t): vale 1 entre -tau/2 y tau/2 y 0 en el resto.
    En los bordes pongo 0.5 (el promedio) para que la señal muestreada
    quede simétrica y la transformada salga más parecida a la teórica.
    """
    x = np.where(np.abs(t) < tau / 2, 1.0, 0.0)
    x[np.isclose(np.abs(t), tau / 2)] = 0.5
    return x


def escalon(t):
    """Escalón unitario u(t): 0 antes de t = 0 y 1 después."""
    return np.where(t > 0, 1.0, np.where(t == 0, 0.5, 0.0))


def escalon_amortiguado(t, b=0.5):
    """
    Escalón con decaimiento exponencial: x(t) = exp(-b t) u(t).
    El escalón puro no tiene transformada de Fourier "ordinaria" porque su
    integral no converge (ejemplo 3.8 del libro). Con b > 0 sí converge, y
    mientras más pequeño sea b más se parece a un escalón.
    """
    return np.where(t >= 0, np.exp(-b * t), 0.0)


def transformada_fourier(x, dt):
    """
    Aproxima la transformada de Fourier continua usando la FFT.

    Pasos:
      - ifftshift: acomodo la señal para que t = 0 quede en la primera
        posición del arreglo. Si no lo hago, la fase sale con un desfase
        que no corresponde a la señal.
      - np.fft.fft: calcula la transformada discreta.
      - fftshift: reordena el resultado para que el cero quede al centro
        y se vean las frecuencias negativas a la izquierda.
      - Multiplico por dt porque la FFT es una suma y la transformada
        de Fourier es una integral (la suma por dt aproxima la integral).

    Regresa w (frecuencia en rad/s) y X (transformada compleja).
    """
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) * dt
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))   # frecuencia en Hz
    w = 2 * np.pi * f                              # la paso a rad/s como en el libro
    return w, X


def fase_grados(X):
    """
    Calcula la fase de X en grados.
    Con números muy pequeños (ruido de redondeo del computador) la fase
    salta sin sentido, así que primero limpio esos valores.
    """
    X = X.copy()
    mx = np.max(np.abs(X))
    # si la parte imaginaria es solo ruido de redondeo la dejo en cero
    X = np.where(np.abs(X.imag) < 1e-9 * mx, X.real + 0j, X)
    fase = np.angle(X, deg=True)
    # donde la magnitud es casi cero la fase no significa nada
    fase[np.abs(X) < 1e-6 * mx] = 0
    return fase


def graficar_tres_paneles(t, x, w, X, titulo, archivo, lim_t, lim_w,
                          X_teorica=None):
    """
    Hace la figura estándar: señal en el tiempo, magnitud y fase.
    Si le paso X_teorica, la dibuja encima con línea punteada para comparar.
    """
    fig, ejes = plt.subplots(3, 1, figsize=(9, 8.5))
    fig.suptitle(titulo, fontsize=13, fontweight="bold", color=VERDE)

    # Panel 1: dominio del tiempo
    ejes[0].plot(t, x, color=VERDE, lw=2)
    ejes[0].set_xlim(lim_t)
    ejes[0].set_xlabel("Tiempo (s)")
    ejes[0].set_ylabel("x(t)")
    ejes[0].set_title("Dominio del tiempo")

    # Panel 2: espectro de magnitud |X(w)|
    ejes[1].plot(w, np.abs(X), color=VERDE, lw=2, label="FFT (numérica)")
    if X_teorica is not None:
        ejes[1].plot(w, np.abs(X_teorica), color=NARANJA, lw=1.5, ls="--",
                     label="Teoría")
        ejes[1].legend()
    ejes[1].set_xlim(lim_w)
    ejes[1].set_xlabel("Frecuencia (rad/s)")
    ejes[1].set_ylabel("|X(w)|")
    ejes[1].set_title("Espectro de magnitud")

    # Panel 3: espectro de fase
    ejes[2].plot(w, fase_grados(X), color=VERDE, lw=2, label="FFT (numérica)")
    if X_teorica is not None:
        ejes[2].plot(w, fase_grados(X_teorica), color=NARANJA, lw=1.5,
                     ls="--", label="Teoría")
        ejes[2].legend()
    ejes[2].set_xlim(lim_w)
    ejes[2].set_xlabel("Frecuencia (rad/s)")
    ejes[2].set_ylabel("Fase de X(w) (grados)")
    ejes[2].set_title("Espectro de fase")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(CARPETA_FIGURAS, archivo), dpi=150)


# ======================================================================
# 2. DEFINICIÓN DE LAS SEÑALES EN EL DOMINIO DEL TIEMPO
# ======================================================================

# Parámetros de muestreo. Uso una ventana de 20 s con 2000 muestras.
dt = 0.01                       # separación entre muestras (s) -> fs = 100 Hz
N = 2000                        # número de muestras
n = np.arange(-N // 2, N // 2)  # índices de -1000 a 999
t = n * dt                      # vector de tiempo de -10 s a 9.99 s (t = 0 incluido)

# Señal 1: pulso rectangular de 1 segundo
tau = 1.0
x_pulso = pulso_rectangular(t, tau)

# Señal 2: escalón unitario y su versión amortiguada (la que sí se puede transformar)
b_dec = 0.5
x_escalon = escalon(t)
x_escalon_amort = escalon_amortiguado(t, b_dec)

# Señal 3: senoidal de 5 Hz (w0 = 2*pi*5 rad/s) con fase de 60 grados (pi/3)
w0 = 2 * np.pi * 5
fase0 = np.pi / 3
x_senoidal = np.cos(w0 * t + fase0)

# ---- Figura 1: las tres señales en el tiempo ----
fig, ejes = plt.subplots(3, 1, figsize=(9, 7))
fig.suptitle("Señales elementales en el dominio del tiempo", fontsize=13,
             fontweight="bold", color=VERDE)

ejes[0].plot(t, x_pulso, color=VERDE, lw=2)
ejes[0].set_xlim(-3, 3)
ejes[0].set_ylim(-0.2, 1.3)
ejes[0].set_title("Pulso rectangular p1(t)")

ejes[1].plot(t, x_escalon, color=VERDE, lw=2, label="Escalón u(t)")
ejes[1].plot(t, x_escalon_amort, color=NARANJA, lw=2, ls="--",
             label="Escalón amortiguado exp(-0.5t)u(t)")
ejes[1].set_xlim(-3, 8)
ejes[1].set_ylim(-0.2, 1.3)
ejes[1].legend(loc="center right")
ejes[1].set_title("Escalón unitario")

ejes[2].plot(t, x_senoidal, color=VERDE, lw=2)
ejes[2].set_xlim(-0.5, 1.5)
ejes[2].set_title("Senoidal cos(2π·5·t + π/3)")
ejes[2].set_xlabel("Tiempo (s)")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(CARPETA_FIGURAS, "fig1_senales_tiempo.png"), dpi=150)


# ======================================================================
# 3. TRANSFORMADA DE FOURIER DE CADA SEÑAL
# ======================================================================

# ---- 3.1 Pulso rectangular ----
w, X_pulso = transformada_fourier(x_pulso, dt)

# Teoría (ecuación 3.39 del libro): X(w) = tau * senc(tau*w / 2pi)
# np.sinc(x) = sen(pi x)/(pi x), que es la misma "senc" del libro
X_pulso_teorica = tau * np.sinc(tau * w / (2 * np.pi))

graficar_tres_paneles(t, x_pulso, w, X_pulso,
                      "Pulso rectangular: tiempo y frecuencia",
                      "fig2_pulso.png", (-3, 3), (-50, 50), X_pulso_teorica)

error_pulso = np.max(np.abs(X_pulso - X_pulso_teorica))
print(f"[Pulso] valor en w = 0: {X_pulso[N // 2].real:.4f} (teoría: {tau})")
print(f"[Pulso] error máximo contra la teoría: {error_pulso:.2e}")

# ---- 3.2 Escalón amortiguado ----
w, X_esc = transformada_fourier(x_escalon_amort, dt)

# Teoría (ejemplo 3.8): X(w) = 1 / (b + jw)
X_esc_teorica = 1 / (b_dec + 1j * w)

graficar_tres_paneles(t, x_escalon_amort, w, X_esc,
                      "Escalón amortiguado exp(-0.5t)u(t): tiempo y frecuencia",
                      "fig3_escalon.png", (-3, 12), (-50, 50), X_esc_teorica)

error_esc = np.max(np.abs(X_esc - X_esc_teorica))
print(f"[Escalón] |X(0)| = {np.abs(X_esc[N // 2]):.4f} (teoría: {1 / b_dec})")
print(f"[Escalón] error máximo contra la teoría: {error_esc:.2e}")

# ---- 3.3 Senoidal ----
w, X_sen = transformada_fourier(x_senoidal, dt)

graficar_tres_paneles(t, x_senoidal, w, X_sen,
                      "Senoidal de 5 Hz: tiempo y frecuencia",
                      "fig4_senoidal.png", (-0.5, 1.5), (-60, 60))

# Busco en qué posición está el pico del espectro (solo frecuencias positivas)
positivas = w > 0
i_pico = np.argmax(np.abs(X_sen) * positivas)
print(f"[Senoidal] pico en w = {w[i_pico]:.3f} rad/s "
      f"(f = {w[i_pico] / (2 * np.pi):.2f} Hz)")
print(f"[Senoidal] magnitud del pico = {np.abs(X_sen[i_pico]):.3f}")
print(f"[Senoidal] fase del pico = {np.angle(X_sen[i_pico], deg=True):.2f} grados")
# La magnitud del pico es A*T/2 = 1*20/2 = 10 porque la ventana dura T = 20 s.
# Una senoidal ideal duraría para siempre y su pico sería un impulso.


# ======================================================================
# 4. VERIFICACIÓN DE PROPIEDADES
# ======================================================================

# Calculo otra vez las transformadas base para usarlas abajo
w, X1 = transformada_fourier(x_pulso, dt)      # pulso
w, X2 = transformada_fourier(x_senoidal, dt)   # senoidal

# ---- 4.1 Linealidad (ecuación 3.40) ----
# Si a*x1 + b*x2 <-> a*X1 + b*X2, da igual sumar primero y transformar
# después, o transformar primero y sumar después.
a_lin, b_lin = 2, 3
x_comb = a_lin * x_pulso + b_lin * x_senoidal
w, X_comb = transformada_fourier(x_comb, dt)    # transformo la suma
X_suma = a_lin * X1 + b_lin * X2                # sumo las transformadas

error_lin = np.max(np.abs(X_comb - X_suma))
print(f"[Linealidad] diferencia máxima entre ambos caminos: {error_lin:.2e}")

fig, ejes = plt.subplots(3, 1, figsize=(9, 8.5))
fig.suptitle("Propiedad de linealidad: x(t) = 2·p1(t) + 3·cos(2π5t + π/3)",
             fontsize=13, fontweight="bold", color=VERDE)
ejes[0].plot(t, x_comb, color=VERDE, lw=1.5)
ejes[0].set_xlim(-3, 3)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].set_ylabel("x(t)")
ejes[0].set_title("Señal combinada en el tiempo")
ejes[1].plot(w, np.abs(X_comb), color=VERDE, lw=2,
             label="Transformada de la suma")
ejes[1].plot(w, np.abs(X_suma), color=NARANJA, lw=1.5, ls="--",
             label="2·X1(w) + 3·X2(w)")
ejes[1].set_xlim(-60, 60)
ejes[1].set_xlabel("Frecuencia (rad/s)")
ejes[1].set_ylabel("Magnitud")
ejes[1].legend()
ejes[1].set_title("Las dos magnitudes coinciden")
ejes[2].semilogy(w, np.abs(X_comb - X_suma) + 1e-20, color=VERDE_CLARO, lw=1.5)
ejes[2].set_xlim(-60, 60)
ejes[2].set_xlabel("Frecuencia (rad/s)")
ejes[2].set_ylabel("Diferencia (escala log)")
ejes[2].set_title("Error entre ambos caminos (solo redondeo del computador)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(CARPETA_FIGURAS, "fig5_linealidad.png"), dpi=150)

# ---- 4.2 Desplazamiento en el tiempo (ecuación 3.41) ----
# x(t - c) <-> X(w) * exp(-j w c): la magnitud no cambia, la fase sí.
c = 2.0                                          # lo retraso 2 segundos
x_desp = pulso_rectangular(t - c, tau)
w, X_desp = transformada_fourier(x_desp, dt)
X_desp_teorica = X1 * np.exp(-1j * w * c)        # lo que dice la propiedad

error_desp = np.max(np.abs(X_desp - X_desp_teorica))
dif_magnitud = np.max(np.abs(np.abs(X_desp) - np.abs(X1)))
print(f"[Desplazamiento] error contra X(w)exp(-jwc): {error_desp:.2e}")
print(f"[Desplazamiento] diferencia entre magnitudes: {dif_magnitud:.2e}")

fig, ejes = plt.subplots(3, 1, figsize=(9, 8.5))
fig.suptitle("Propiedad de desplazamiento en el tiempo (c = 2 s)",
             fontsize=13, fontweight="bold", color=VERDE)
ejes[0].plot(t, x_pulso, color=VERDE, lw=2, label="p1(t)")
ejes[0].plot(t, x_desp, color=NARANJA, lw=2, ls="--", label="p1(t - 2)")
ejes[0].set_xlim(-3, 5)
ejes[0].set_ylim(-0.2, 1.3)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend()
ejes[0].set_title("Dominio del tiempo")
ejes[1].plot(w, np.abs(X1), color=VERDE, lw=2, label="|X| original")
ejes[1].plot(w, np.abs(X_desp), color=NARANJA, lw=1.5, ls="--",
             label="|X| desplazada")
ejes[1].set_xlim(-20, 20)
ejes[1].set_xlabel("Frecuencia (rad/s)")
ejes[1].set_ylabel("Magnitud")
ejes[1].legend()
ejes[1].set_title("La magnitud queda igual")
ejes[2].plot(w, fase_grados(X1), color=VERDE, lw=2, label="Fase original")
ejes[2].plot(w, fase_grados(X_desp), color=NARANJA, lw=1.5,
             label="Fase desplazada")
ejes[2].set_xlim(-20, 20)
ejes[2].set_xlabel("Frecuencia (rad/s)")
ejes[2].set_ylabel("Fase (grados)")
ejes[2].legend(loc="upper right")
ejes[2].set_title("La fase cambia (aparece el término -w·c)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(CARPETA_FIGURAS, "fig6_desplazamiento.png"), dpi=150)

# ---- 4.3 Escalamiento (ecuación 3.43) ----
# x(at) <-> (1/a) X(w/a). Con a = 2 el pulso se comprime en el tiempo
# y su espectro se ensancha en frecuencia.
a_esc = 2
x_esc = pulso_rectangular(a_esc * t, tau)        # p1(2t): dura 0.5 s
w, X_esc2 = transformada_fourier(x_esc, dt)

# Construyo (1/a) X(w/a) interpolando el espectro original
X_esc_teorica2 = np.interp(w / a_esc, w, X1.real) / a_esc

zona = np.abs(w) <= 60
error_esc2 = np.max(np.abs(X_esc2.real[zona] - X_esc_teorica2[zona]))
print(f"[Escalamiento] error contra (1/a)X(w/a): {error_esc2:.2e}")
print(f"[Escalamiento] primer cero del pulso original: {2 * np.pi / tau:.2f} rad/s")
print(f"[Escalamiento] primer cero del pulso comprimido: "
      f"{2 * np.pi / (tau / a_esc):.2f} rad/s")

fig, ejes = plt.subplots(2, 1, figsize=(9, 6.5))
fig.suptitle("Propiedad de escalamiento: p1(t) contra p1(2t)", fontsize=13,
             fontweight="bold", color=VERDE)
ejes[0].plot(t, x_pulso, color=VERDE, lw=2, label="p1(t)  (dura 1 s)")
ejes[0].plot(t, x_esc, color=NARANJA, lw=2, label="p1(2t)  (dura 0.5 s)")
ejes[0].set_xlim(-2, 2)
ejes[0].set_ylim(-0.2, 1.3)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend()
ejes[0].set_title("Compresión en el tiempo")
ejes[1].plot(w, X1.real, color=VERDE, lw=2, label="X(w) del pulso de 1 s")
ejes[1].plot(w, X_esc2.real, color=NARANJA, lw=2,
             label="X(w) del pulso de 0.5 s")
ejes[1].plot(w, X_esc_teorica2, color=GRIS, lw=1.2, ls=":",
             label="(1/2)·X(w/2) teórica")
ejes[1].set_xlim(-50, 50)
ejes[1].set_xlabel("Frecuencia (rad/s)")
ejes[1].set_ylabel("X(w)")
ejes[1].legend()
ejes[1].set_title("Expansión en frecuencia (el espectro se ensancha y baja a la mitad)")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(CARPETA_FIGURAS, "fig7_escalamiento.png"), dpi=150)


# ======================================================================
# 5. APLICACIÓN REAL: MODULACIÓN DE AMPLITUD (RADIO AM)
# ======================================================================
# Para transmitir una señal x(t) se multiplica por una portadora cos(wc t)
# (sección 3.8 del libro). Por la propiedad de modulación (ec. 3.52 y 3.51)
# el espectro de x(t) se copia alrededor de +wc y de -wc.
#
# Aquí uso una frecuencia de muestreo más alta (1000 Hz) porque la portadora
# de 50 Hz necesita más muestras por segundo que las señales de arriba.

dt_am = 0.001
N_am = 10000                                     # ventana de 10 s
t_am = np.arange(-N_am // 2, N_am // 2) * dt_am

f_mensaje = 2.0       # "mensaje" de 2 Hz (hace el papel del audio)
f_portadora = 50.0    # portadora de 50 Hz (en radio real serían kHz o MHz)

mensaje = np.cos(2 * np.pi * f_mensaje * t_am)
portadora = np.cos(2 * np.pi * f_portadora * t_am)
modulada = mensaje * portadora                   # AM de doble banda lateral (DSB-SC)

w_am, M = transformada_fourier(mensaje, dt_am)
w_am, S = transformada_fourier(modulada, dt_am)
f_am = w_am / (2 * np.pi)                        # aquí sí grafico en Hz


def frecuencias_pico(f, X, fraccion=0.3):
    """Devuelve las frecuencias positivas (Hz) donde el espectro tiene picos."""
    mag = np.abs(X)
    return f[(f > 0) & (mag > fraccion * mag.max())]


print(f"[AM] picos del mensaje (Hz): {frecuencias_pico(f_am, M)}")
print(f"[AM] picos de la señal modulada (Hz): {frecuencias_pico(f_am, S)}")
print(f"[AM] magnitud pico del mensaje: {np.abs(M).max():.2f} | "
      f"de la modulada: {np.abs(S).max():.2f}")

fig, ejes = plt.subplots(3, 1, figsize=(9, 8.5))
fig.suptitle("Modulación de amplitud: mensaje de 2 Hz y portadora de 50 Hz",
             fontsize=13, fontweight="bold", color=VERDE)
ejes[0].plot(t_am, modulada, color=VERDE, lw=1, label="s(t) = x(t)·cos(2π50t)")
ejes[0].plot(t_am, mensaje, color=NARANJA, lw=2, label="x(t) (mensaje)")
ejes[0].plot(t_am, -mensaje, color=NARANJA, lw=1, ls="--")
ejes[0].set_xlim(0, 2)
ejes[0].set_xlabel("Tiempo (s)")
ejes[0].legend(loc="upper right")
ejes[0].set_title("Dominio del tiempo (la envolvente sigue al mensaje)")
ejes[1].plot(f_am, np.abs(M), color=VERDE, lw=2)
ejes[1].set_xlim(-70, 70)
ejes[1].set_xlabel("Frecuencia (Hz)")
ejes[1].set_ylabel("|X|")
ejes[1].set_title("Espectro del mensaje: picos en ±2 Hz")
ejes[2].plot(f_am, np.abs(S), color=VERDE, lw=2)
ejes[2].set_xlim(-70, 70)
ejes[2].set_xlabel("Frecuencia (Hz)")
ejes[2].set_ylabel("|S|")
ejes[2].set_title("Espectro de la señal modulada: picos en ±48 Hz y ±52 Hz")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(CARPETA_FIGURAS, "fig8_modulacion_am.png"), dpi=150)


# ======================================================================
# 6. FINAL
# ======================================================================
print("\nListo. Las imágenes quedaron guardadas en la carpeta:", CARPETA_FIGURAS)

if MOSTRAR:
    plt.show()
