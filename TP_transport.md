# TP transport d'un traceur à deux dimensions

Utilisez le code transport2d pour illustrer ces concepts

  - stabilité d'un schéma
  - dispersion vs. diffusion
  - ordre d'un schéma et précision de la solution
  - mixing vs stirring
  - schémas upwind vs centrés
  - interpolation linéaire vs nonlinéaire (WENO)
  - diffusion implicite
  - conservation globale
  - reversibilité

# Le code intègre l'équation de transport

  - écrite sous forme flux
  - le traceur 'q' est considéré comme un volume fini = la valeur
    discrétisée est la moyenne de la maille, pas la valeur au centre
    de la maille
  - en utilisant une discrétisation sur une grille "C"
  - plusieurs schémas en temps sont proposés: RK3 pour les
    interpolations WENO, LF+Asselin = le schéma en temps du code NEMO,
    LF-AM3 = le schéma en temps du code CROCO.
  - le flux aux interfaces est calculé par une interpolation d'ordre
    donné -> paramètre flxorder = 1, 2, 3, 4, 5
  - pour les ordres impairs, l'interpolation peut être nonlinéaire via
    WENO


# combos intéressants

  - Leap-Frog + bodyrotation + 2nd order -> dispersion
  - Leap-Frog + bodyrotation + 3rd order -> instable
  - LFAMF + bodyrotation + 3rd order -> diffusion numérique + dispersion
  - LFAMF + bodyrotation + 5rd order -> higher order
  - RK3 + bodyrotation + 5rd order -> LFAM3 looks quite similar to RK3
  - RK3 + bodyrotation + 5rd order + WENO -> positivité
  - RK3 + bodyrotation + 5rd order + WENO + quadripole -> stirring
  - Leap-Frog + bodyrotation + 2nd order + quadripole -> stirring + dispersion
  - Leap-Frog + bodyrotation + 2nd order + backward -> reversibilité
  - RK3 + bodyrotation + 5rd order + WENO -> irreversibilité

# figures pour l'analyse

  - vérifier la conservation globale de q -> formulation flux
  - observer la conservation ou la destruction de variance de q^2 -> dissipation
  - observer l'évolution de l'histogramme de q entre initial et final
