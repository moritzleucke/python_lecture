"""
Author: Jannis Kockläuner
Datum: 09.06.23

Numerische Zeitentwicklung eines Gausschen Wellenpakets
im Doppelmuldenpotential
Der mittlere Ort für t=0 wird mittels Linksklick bestimmt
"""

import quantenmechanik as qm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def GaussPaket(x, x0, p0, dx, heff):
    """
    Amplitude eines Gauss Paket in Ortsdarstellung
    Parameter:
    x: diskrete Ortskoordinaten, Array 
    x0: mittlerer Ort, float
    dx: Breite d. Wellenpakets, float
    p0: Impuls, float
    h_eff: effektiver parameter, float
    """
    
    pre = 1 / ((2 * np.pi * (dx)**2) ** 1 / 4)     # Normierungsfaktor  
    ampli = np.exp(-(x - x0) ** 2 / (4 * dx ** 2))   # reele Amplitude
    phase = np.exp(1j * p0 * x / heff)               # komplexe Phase
    return pre * ampli * phase


def Entwicklung(dx, psi, vecs, vals):
    """
    Entiwckelt eine Wellenfunktion psi in Eigenfunktionen c
    Parameter: 
    x: diskrete Ortskoordinaten, Array shape: (Nx)
    psi: Amplituden Psi(x), Array shape: (Nx)
    c: Eigenfunktionen in denen entwickelt wird, Array shape: (Nx, N)
    Returns: N Entwicklungskoeffizienten 
    Returns: entwicklungskoeffizienten, Energie
    """
    c_i = np.dot(np.conjugate(psi), vecs) * dx   # Entwicklungskoeffizienten 
    e = np.dot(np.conjugate(vals), abs(c_i)**2)  # <psi|H|psi> in EV basis
    return c_i, e


def DoppelPot(x, A):
    """
    Doppelmuldenpotential am Punkt x
    Parameter:
    x: diskrete Orskoordinaten, Array
    A: relative Verschiebbung d. Potentialtöpfe, float 
    """
    return x ** 4 - x ** 2 - A * x


def Zeitentwicklung(vals, vecs, c0, t, heff): 
    """
    Berechnet die Zeitenwicklung eines Zustands in der Eigenbasis
    Parameter: 
    vals: Eigenwerte, Array (N)
    vecs: Eigenvektoren, Array (Nx, N)
    c0: Entwickungskoeffizienten, Array (N)
    t: diskrete Zeitpunte, Array (Nt)
    heff: effektiver parameter d. Modells, float
    Returns: psi(t) als Array(Nt, Nx)
    """
    e_t = np.outer(vals, t)
    phase = np.exp(- 1j * e_t / heff)         # Shape: (N, Nt)
    vecs = np.expand_dims(vecs, axis=-1)      # 3D elementweise Multiplikation
    ct = phase * vecs                         # Shape: (Nt, N, Nx)
    psi = np.einsum("i, jik->jk", c0, ct.T)   # Shape: (Nt, Nx)
    return psi


def Plot_Zeitentwicklung(event, x, t, vals, vecs, gauss, ax, heff, scale):
    """
    Plottet Zeitentwicklung eines Wellenpaket zentriert am Mausklick
    Parameter: 
    event: Mausklick event
    x: diskrete Ortskoordinaten, Array
    t: diskrete Zeitunkte, Array
    vals: Eigenwerte d. Doppelmuldenpotentials, Array (N_c)
    vecs: Eigenvektoren d. Doppelmuldenpotentials, Array(N_x, N_c)
    gauss: Gaussches Wellenpaket, Callable
    ax: Plot Axis
    heff: effektives Parameter d. Modells, float
    scale: Skalierung der Wellenfunktion
    """

    mode = event.canvas.toolbar.mode  # Prüfe Zoommodus
    dx = x[1] - x[0]                  # Diskretisierungsschritte

    # Wenn Linksklick im Plotbereich und kein Zoom
    if event.button == 1 and event.inaxes and mode == "": 
        psi0 = gauss(x, x0=event.xdata)            # Start Wellenfunktion
        psi0 /= np.sqrt(np.sum(abs(psi0) ** 2))    # Normalization
        # Normalisierung kompatibel mit Vektoren aus quantenmechanik.py
        psi0 /= np.sqrt(dx)                       
        # Entwickle psi0 in eigenvektoren 
        c_i, e = Entwicklung(dx, psi0, vecs, vals)   
        # Prüfe Qualität der Entwicklung
        d_psi = dx * (np.sum(c_i * vecs, axis=1) - psi0.conjugate())
        d_psi = np.linalg.norm(d_psi)
        print(f"Absoluter Fehler der entwickelten WF für t=0 beträgt {d_psi}")
        # Zeitentwicklung psi(t)
        psi_t = Zeitentwicklung(vals, vecs, c_i, t, heff)       
        ax.hlines(e, min(x), max(x))                        # Zeige Eigenwert
        plot = ax.plot(x, scale * abs(psi0).real ** 2 + e, lw=3)  # psi(t=0)
        
        # Plotte psi(t) dynamisch 
        for psi in psi_t:
            plot[0].set_ydata(scale * abs(psi).real ** 2 + e)   # neue y-Daten 
            event.canvas.flush_events()
            event.canvas.draw()
        

def main():
    """
    Zeitentwicklung eines Gausschen Wellenpakets 
    Das Zentrum des Pakets für t = 0 wird mittels Linksklick gewählt
    Zum Vergleich werden die numerisch bestimmten Eigenfunktionen 
    mit E_n < 0.15 dargestellt
    """
    # Parameter 
    A = 0.05      # Tiefe asymmetrische Doppelmulde
    heff = 0.06   # effektver parameter eingeitenloses Potential
    p0 = 0        # Gauss Parameter, mittlerer Impuls 
    x0 = 0        # Gauss Paramter, mittlerer Ort
    dx = 0.1      # Gauss Paramter, weite d. Pakets
    N_x = 1000    # Diskretisierung: Anzahl Ortspunkte
    xmin = -3     # Diskretisierung: max/min Werte
    xmax = 3 
    N_c = 50      # Anzahl Eigenvektoren fpr numerische Entwicklung
    N_t = 1000    # Anzahl diskrete Zeitpunkte
    tmax = 100    # Ende d. Zeitentwicklung
    emax = 0.15   # Threshold für Eigenwerte im Plot
    scale = 0.02  # Skalierung psi(x), größer als Eigenfunktionen

    # Ausgabe der relevanten parameter 
    print(__doc__)
    print(f"Doppelmuldenpotential: V(x) = x^4 - x^2 - {A}*x") 
    print(f"Effektiver Parameter heff: {heff}")
    print(f"Gausspaket: \n dx = {dx} \n p0 = {p0}")
    print(f"Zeitentwicklung: \n"
          f"Eigenzustände in numerischer Entwicklung = {N_c}\n" 
          f"tmax = {tmax}")

    # Setze parameter: Gauss Paket und Potential
    x = qm.diskretisierung(xmin, xmax, N_x)   # Diskrete Ortskoordinaten
    t = np.linspace(0, tmax, N_t)             # Diskrete Zeitpunkte
    gauss = partial(GaussPaket, dx=dx, heff=heff, p0=p0)   # Gauss Paket 
    pot = partial(DoppelPot, A=A)   # Doppelmuldenpotential
    # Löse Schroedingergleichung für Doppelmuldenpotential numerisch
    vals, vecs = qm.diagonalisierung(heff, x, pot)  
    vals = vals[:N_c]
    vecs = vecs[:, :N_c]
    # Initialisiere plot
    fig, ax = plt.subplots()
    # Plotte Eigenfunktionen d. Doppelmuldenpotentials
    qm.plot_eigenfunktionen(ax, vals, vecs, x, pot, Emax=emax)
    # Setzt Parameter für klick Funktion
    klick_f = partial(Plot_Zeitentwicklung, 
                      x=x, 
                      t=t, 
                      vals=vals,
                      vecs=vecs,
                      gauss=gauss,
                      ax=ax,
                      heff=heff,
                      scale=scale)
    # Verknüpfe mit Mausklick
    fig.canvas.mpl_connect("button_press_event", klick_f)
    plt.show()
    

if __name__ == "__main__":
    main()
    

"""
a) 
Start im Minimum: 
Bewegung des Wellenpakets ähnlich zum harmonischen Oszialltor,
das Wellenpaket bleibt näherungsweise Gauss-förmig und die maximale Amplitude 
ist immer im Bereich des Minimums
Das Wellenpaket ist lokalisiert im Bereich der Minima und wird vin wenigen, 
energetisch niedrigen Eigenfunktionen gut beschrieben
Start im Maximum: 
Wellenpaket zerfliesst, delokalisiert zwischen beiden Minima 
Die Ausdehnung des Wellenpakets ist wesentlich größer, dh. es müssen viele 
Eigenfunktionen für die Entwicklung verwendet werden
b) p0=0.3
Start im Minimum: Wellenpaket zerfließt mit der Zeit, der dominante Eigenvektor
d. Entwicklung wechselt mit der Zeit aufgrund der komplexen Phase, 
das Paket tunnelt zwischen Minima hin und her, allerdings mit kleiner Amplitude
Start im Maximum: Wellenpaket zerfließt, delokalisiert zwischen beiden Minma
c) A=0, tmax = 10**4
Wenn das Wellenpaket in einem der Minima gestartet wird, 
tunnelt es mit der Zeit in den Bereich des anderen Minimums, 
bis die Ampltiude komplett im Bereich des zweiten Minimums liegt
Demnach oscilliert das Wellenpaket, unabhängig von der Startposition, 
zwischen beiden Minima 
"""
